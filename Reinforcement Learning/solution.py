import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.utils import seeding
from utils import ReplayBuffer, get_env, run_episode


class MLP(nn.Module):
    '''
    A simple ReLU MLP constructed from a list of layer width
    '''
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(in_size, out_size))
            if i < len(sizes) - 2:
                layers.append(nn.LayerNorm(out_size)) #normalize to stabilize
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class Critic(nn.Module):
    '''
    TD3 Critic
    '''
    def __init__(self, obs_size, action_size, num_layers, num_units):
        super().__init__()
        self.net = MLP([obs_size + action_size] + ([num_units] * num_layers) + [1])

    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=-1))


class Actor(nn.Module):
    '''
    TD3 Actor
    Output in [-1, 1] (squashed by Tanh)
    '''
    def __init__(self, action_low, action_high, obs_size, action_size, num_layers, num_units):
        super().__init__()
        self.net = MLP([obs_size] + ([num_units] * num_layers) + [action_size])
        self.action_scale = (action_high - action_low) / 2
        self.action_bias = (action_high + action_low) / 2

    def forward(self, x):
        x = self.net(x)
        x = torch.tanh(x)
        return x * self.action_scale + self.action_bias


class Agent:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    buffer_size: int = 100_000 

    #########################################################################
    # TODO: store and tune hyperparameters here
    num_layers: int = 4
    num_units: int = 256
    batch_size: int = 256
    gamma: float = 0.99           # MDP Discount factor
    tau: float = 0.005            # Polyak averaging
    learning_rate_q: float = 3e-4
    learning_rate_pi: float = 3e-4
    policy_noise: float = 0.2     # Noise added to target actor (smoothing)
    noise_clip: float = 0.5       # Noise clip 
    policy_freq: int = 2          # Update frequency
    exploration_noise: float = 0.1 
    #########################################################################

    def __init__(self, env):
        self.obs_size = np.prod(env.observation_space.shape)
        self.action_size = np.prod(env.action_space.shape)
        self.action_low = torch.tensor(env.action_space.low).float().to(self.device)
        self.action_high = torch.tensor(env.action_space.high).float().to(self.device)
        self.action_scale = (self.action_high - self.action_low) / 2
        self.action_bias = (self.action_high + self.action_low) / 2

        # 1. Initialize Actor & Target
        self.pi = Actor(self.action_low, self.action_high, self.obs_size, self.action_size, self.num_layers, self.num_units).to(self.device)
        self.pi_target = Actor(self.action_low, self.action_high, self.obs_size, self.action_size, self.num_layers, self.num_units).to(self.device)
        self.pi_target.load_state_dict(self.pi.state_dict())

        # 2. Initialize Twin Critics & Targets
        self.q1 = Critic(self.obs_size, self.action_size, self.num_layers, self.num_units).to(self.device)
        self.q2 = Critic(self.obs_size, self.action_size, self.num_layers, self.num_units).to(self.device)
        
        self.q1_target = Critic(self.obs_size, self.action_size, self.num_layers, self.num_units).to(self.device)
        self.q2_target = Critic(self.obs_size, self.action_size, self.num_layers, self.num_units).to(self.device)
        
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # 3. Optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=self.learning_rate_q)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=self.learning_rate_q)
        self.pi_optimizer = optim.Adam(self.pi.parameters(), lr=self.learning_rate_pi)

        self._init_weights(self.pi)
        self._init_weights(self.q1)
        self._init_weights(self.q2)
        self.pi_target.load_state_dict(self.pi.state_dict())
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.buffer = ReplayBuffer(self.buffer_size, self.obs_size, self.action_size, self.device)
        self.train_step = 0

    def _init_weights(self, module):
        '''Initialize weights'''
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
        
        if hasattr(module, 'net'):
            last_layer = module.net.layers[-1]
            if isinstance(last_layer, nn.Linear):
                last_layer.weight.data.mul_(0.001)
                last_layer.bias.data.mul_(0.001)

    def train(self):
        self.train_step += 1
        obs, action, next_obs, done, reward = self.buffer.sample(self.batch_size)
        
        if reward.dim() == 1: reward = reward.unsqueeze(-1)
        if done.dim() == 1: done = done.unsqueeze(-1)

        with torch.no_grad():
            # Target Policy Smoothing
            next_action = self.pi_target(next_obs)

            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (next_action + noise).clamp(self.action_low, self.action_high)

            target_q1 = self.q1_target(next_obs, next_action)
            target_q2 = self.q2_target(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2)
            
            # Bellman Equation
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1 = self.q1(obs, action)
        current_q2 = self.q2(obs, action)

        loss_q = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        loss_q.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # Delayed Policy Updates
        if self.train_step % self.policy_freq == 0:
            for p in self.q1.parameters(): p.requires_grad = False

            actor_loss = -self.q1(obs, self.pi(obs)).mean()
            
            self.pi_optimizer.zero_grad()
            actor_loss.backward()
            self.pi_optimizer.step()

            for p in self.q1.parameters(): p.requires_grad = True

            # Polyak Averaging
            with torch.no_grad():
                for param, target_param in zip(self.pi.parameters(), self.pi_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def get_action(self, obs, train):

        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.pi(obs)
            
            if train:
                noise = torch.randn_like(action) * self.exploration_noise
                action = (action + noise).clamp(self.action_low, self.action_high)
            
            return action.cpu().numpy().flatten()

    def store(self, transition):
        obs, action, reward, next_obs, terminated = transition
        self.buffer.store(obs, next_obs, action, reward, terminated)


# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    WARMUP_EPISODES = 10  # initial episodes of uniform exploration
    TRAIN_EPISODES = 50  # interactive episodes
    TEST_EPISODES = 300  # evaluation episodes
    save_video = True
    verbose = True
    seeds = np.arange(10)  # seeds for public evaluation

    start = time.time()
    print(f'Running public evaluation.') 
    test_returns = {k: [] for k in seeds}

    for seed in seeds:

        # seeding to ensure determinism
        seed = int(seed)
        for fn in [random.seed, np.random.seed, torch.manual_seed]:
            fn(seed)
        torch.backends.cudnn.deterministic = True

        env = get_env()
        env.action_space.seed(seed)
        env.np_random, _ = seeding.np_random(seed)

        agent = Agent(env)

        for _ in range(WARMUP_EPISODES):
            run_episode(env, agent, mode='warmup', verbose=verbose, rec=False)

        for _ in range(TRAIN_EPISODES):
            run_episode(env, agent, mode='train', verbose=verbose, rec=False)

        for n_ep in range(TEST_EPISODES):
            video_rec = (save_video and n_ep == TEST_EPISODES - 1)  # only record last episode
            with torch.no_grad():
                episode_return = run_episode(env, agent, mode='test', verbose=verbose, rec=video_rec)
            test_returns[seed].append(episode_return)

    avg_test_return = np.mean([np.mean(v) for v in test_returns.values()])
    within_seeds_deviation = np.mean([np.std(v) for v in test_returns.values()])
    across_seeds_deviation = np.std([np.mean(v) for v in test_returns.values()])
    print(f'Score for public evaluation: {avg_test_return}')
    print(f'Deviation within seeds: {within_seeds_deviation}')
    print(f'Deviation across seeds: {across_seeds_deviation}')

    print("Time :", (time.time() - start)/60, "min")
