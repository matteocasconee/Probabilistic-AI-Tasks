import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from scipy.stats import norm
import matplotlib.pyplot as plt
# import additional ...


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        

        # Observed data
        self.X = None           # shape (n_samples, D)
        self.y_f = None         # shape (n_samples,)
        self.y_v = None          # ricorda il prior!
        self.y_v_prior_mean = 0.0
        
        # Hyperparameters
        self.sigma_f = 0.15
        self.sigma_v = 0.0001

        # Constraint threshold and Lagrangian relaxation for constraint violation
        self.kappa = SAFETY_THRESHOLD
        self.mylambda = 100

        # Best safe observed (not predicted)
        self.best_safe_x = None
        self.best_safe_y = None

        # Kernel for f and v: 
        kernel_f = Matern(length_scale=1,nu=2.5)
        kernel_v = DotProduct(sigma_0=1.0)**2 + Matern(length_scale=0.5, nu=2.5)

        # GaussianProcessRegressor: alpha = noise variance 
        self.gp_f = GaussianProcessRegressor(kernel=kernel_f,
                                             alpha=self.sigma_f ** 2,
                                             normalize_y=True)

        self.gp_v = GaussianProcessRegressor(kernel=kernel_v,
                                             alpha=self.sigma_v ** 2,
                                             normalize_y=True)
        pass

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.
        return self.optimize_acquisition_function()

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick the best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])
        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.
        mean_f, std_f = self.gp_f.predict(x, return_std=True)
        mean_v, std_v = self.gp_v.predict(x, return_std=True)
        P_sub_threshold=norm.cdf((self.kappa - mean_v) / std_v)
        if self.best_safe_y is None:
            return P_sub_threshold #P of being below the threshold
        penalty=self.mylambda*np.maximum(mean_v-self.kappa,0)
        beta=1.05
        UCB=mean_f+beta*std_f
        return UCB*P_sub_threshold-penalty


    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float or np.ndarray
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.
        x = np.asarray(x).reshape(1, -1)

        if self.X is None:
            self.X = x
            self.y_f = np.array([f])
            self.y_v = np.array([v-self.y_v_prior_mean])
        else:
            self.X = np.vstack([self.X, x])
            self.y_f = np.append(self.y_f, f)
            self.y_v = np.append(self.y_v, v-self.y_v_prior_mean)

        if v < self.kappa:
            if (self.best_safe_y is None) or (float(f) > self.best_safe_y):
                self.best_safe_y = float(f)
                self.best_safe_x = x
        self.gp_f.fit(self.X, self.y_f) 
        self.gp_v.fit(self.X, self.y_v) 

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        
        if self.best_safe_x is not None:
            return self.best_safe_x
        num_pts=10000
        set=np.linspace(DOMAIN[0,0],DOMAIN[0,1],num_pts).reshape(-1,1)
        mean_v, std_v = self.gp_v.predict(set, return_std=True)
        P_safe = norm.cdf((self.kappa - mean_v) / std_v)
        return set[np.argmax(P_safe)][0]

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        x_grid = np.linspace(DOMAIN[0, 0], DOMAIN[0, 1], 500).reshape(-1, 1)
    
        # Calculates mean and std for logP
        mean_f, std_f = self.gp_f.predict(x_grid, return_std=True)
        
        # Calculates mean and std for SA 
        mean_v, std_v = self.gp_v.predict(x_grid, return_std=True)
        
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.fill_between(x_grid.flatten(), mean_f - 2 * std_f, mean_f + 2 * std_f, alpha=0.2, color='blue', label="95% CI")
        plt.plot(x_grid, mean_f, 'b-', label="GP mean f")
        plt.scatter(self.X, self.y_f, c='blue', marker='x', label="Observed f")
        if self.best_safe_x is not None:
            plt.scatter(self.best_safe_x, self.best_safe_y, c='green', s=100, label="Best safe point")
        plt.xlabel("x")
        plt.ylabel("f(x) (logP)")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.fill_between(x_grid.flatten(), mean_v - 2 * std_v, mean_v + 2 * std_v, alpha=0.2, color='orange', label="95% CI")
        plt.plot(x_grid, mean_v, 'r-', label="GP mean v")
        plt.axhline(self.kappa, color='black', linestyle='--', label=f"Threshold = {self.kappa}")
        plt.scatter(self.X, self.y_v + self.y_v_prior_mean, c='red', marker='x', label="Observed v")  # Ricorda che v è centrato
        plt.xlabel("x")
        plt.ylabel("v(x) (SA)")
        plt.legend()

        if plot_recommendation:
            x_next = self.next_recommendation()
            plt.subplot(1, 2, 1)
            plt.axvline(x_next, color='purple', linestyle='--', label="Next recommendation")
            plt.subplot(1, 2, 2)
            plt.axvline(x_next, color='purple', linestyle='--')

        plt.tight_layout()
        plt.show()



# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.randn()
        cost_val = v(x) + np.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
