import torch 
import numpy as np
from typing import Tuple

from carla_drl.lane_following.model import ActorCritic


class Buffer:
    def __init__(self):
         # Batch data
        self.ss_obs = []  
        self.depth_obs = []  
        self.nav_obs = []  
        self.actions = []         
        self.log_probs = []     
        self.rewards = []         
        self.dones = []

    def clear(self):
        del self.ss_obs[:]    
        del self.depth_obs[:]    
        del self.nav_obs[:]    
        del self.actions[:]        
        del self.log_probs[:]      
        del self.rewards[:]
        del self.dones[:]

class PPOAgent:
    def __init__(self, 
                 ss_channels: int = 128,
                 depth_channels: int = 128,
                 navigation_dim: int = 5,
                 action_dim: int = 2,
                 dropout: float = 0.2,
                 learning_rate: float = 1e-4,
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 epsilon: float = 0.2,
                 epoch_n: int = 30,
                 device: str = 'cpu'
                 ):
        self.ss_channels = ss_channels
        self.depth_channels = depth_channels
        self.action_dim = action_dim

        self.batch_size = batch_size
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epoch_n = epoch_n
        self.device = device

        self.memory = Buffer()

        self.policy = ActorCritic(
            ss_channels=self.ss_channels,
            depth_channels=self.depth_channels,
            nav_dim=navigation_dim,
            action_dim=self.action_dim,
            dropout=dropout
        ).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.ss_head.parameters(), 'lr': self.lr},
            {'params': self.policy.depth_head.parameters(), 'lr': self.lr},
            {'params': self.policy.actor.parameters(), 'lr': self.lr},
            {'params': self.policy.critic.parameters(), 'lr': self.lr},
        ])

    def get_action(self, observation: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], train: bool = True) -> np.ndarray:

        with torch.no_grad():
            observation = tuple(obs.to(self.device) for obs in observation)
            action, logprob = self.policy.get_action_and_logprobs(observation)
        if train:
            self.memory.ss_obs.append(observation[0])
            self.memory.depth_obs.append(observation[1])
            self.memory.nav_obs.append(observation[2])
            self.memory.actions.append(action)
            self.memory.log_probs.append(logprob)

        return action.detach().cpu().numpy().flatten()

    def learn(self):
        
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # convert list to tensor
        old_ss_states = torch.squeeze(torch.stack(self.memory.ss_obs, dim=0)).detach().to(self.device)
        old_depth_states = torch.squeeze(torch.stack(self.memory.depth_obs, dim=0)).detach().to(self.device)
        old_nav_states = torch.squeeze(torch.stack(self.memory.nav_obs, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.memory.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.memory.log_probs, dim=0)).detach().to(self.device)

        for epoch in range(self.epoch_n):
            idxs = np.random.permutation(old_nav_states.shape[0])
            for i in range(0, old_nav_states.shape[0], self.batch_size):
                idx = idxs[i:i + self.batch_size]
                b_actions, b_rewards, b_old_logprobs = old_actions[idx], rewards[idx], old_logprobs[idx]
                b_states = (old_ss_states[idx], old_depth_states[idx], old_nav_states[idx])

                b_log_probs, b_values, b_dist_entropy = self.policy.evaluate(b_states, b_actions)
                b_values = torch.squeeze(b_values)

                ratio = torch.exp(b_log_probs - b_old_logprobs)

                b_advantages = b_rewards - b_values.detach()

                pi_loss1 = ratio * b_advantages
                pi_loss2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * b_advantages

                pi_loss = -torch.min(pi_loss1, pi_loss2)
                v_loss = 0.5 * (b_advantages ** 2)

                loss = (pi_loss + v_loss).mean()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

        self.memory.clear()

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        self.policy.load_state_dict(torch.load(path, map_location='cpu'))