from typing import Tuple
import math

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class ActorCritic(nn.Module):
    def __init__(self, ss_channels: Tuple[int], depth_channels: Tuple[int], nav_dim: int,
                 action_dim: int, dropout: float = 0.):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim
        
        # Define the first head for input shape (B, 128, 16, 30)
        self.ss_head = nn.Sequential(
            nn.Conv2d(in_channels=ss_channels, out_channels=64, kernel_size=3, stride=1, padding=1),  # Output: (B, 64, 16, 30)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (B, 64, 8, 15)
            nn.BatchNorm2d(64),  # Batch normalization
            
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),  # Output: (B, 32, 8, 15)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (B, 32, 4, 7)
            nn.BatchNorm2d(32)  # Batch normalization
        )

        # Define the second head for input shape (B, 128, 7, 12) with max pooling
        self.depth_head = nn.Sequential(
            nn.Conv2d(in_channels=depth_channels, out_channels=64, kernel_size=3, stride=1, padding=1),  # Output: (B, 64, 7, 12)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: (B, 64, 3, 6)
            nn.BatchNorm2d(64),  # Batch normalization
            
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),  # Output: (B, 32, 3, 6)
            nn.ReLU(),
            nn.BatchNorm2d(32)  # Batch normalization
        )

        # Define the final linear layers after concatenation in a sequential model
        self.actor = nn.Sequential(
            nn.Linear(32 * 4 * 7 + 32 * 3 * 6 + nav_dim, 128),  # Adjusted input size based on concatenated output
            nn.ReLU(),
            nn.Dropout(dropout),  # Dropout layer
            nn.Linear(128, 2 * action_dim)  # Output layer
        )

        self.critic = nn.Sequential(
            nn.Linear(32 * 4 * 7 + 32 * 3 * 6 + nav_dim, 128),  # Adjusted input size based on concatenated output
            nn.ReLU(),
            nn.Dropout(dropout),  # Dropout layer
            nn.Linear(128, 1)  # Output layer
        )

        self.init_weights()  # Call the weight initialization method

    def init_weights(self):
        for layer in self.ss_head:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))  # He initialization
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        for layer in self.depth_head:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))  # He initialization
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        # Initialize fully connected layers
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                nn.init.constant_(layer.bias, 0)
        for layer in self.critic:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                nn.init.constant_(layer.bias, 0)

    def _get_features(self, ss_obs: torch.Tensor, depth_obs: torch.Tensor, nav_obs: torch.Tensor) -> torch.Tensor:
        # Forward pass through both heads
        ss_obs = self.ss_head(ss_obs)
        ss_obs = ss_obs.view(ss_obs.size(0), -1)  # Flatten
        depth_obs = self.depth_head(depth_obs)
        depth_obs = depth_obs.view(depth_obs.size(0), -1)  # Flatten
        
        # Concatenate features from both heads
        x = torch.cat((ss_obs, depth_obs, nav_obs), dim=1)
        
        return x

    def get_value(self, observation: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        ss_obs, depth_obs, nav_obs = observation
        features = self._get_features(ss_obs, depth_obs, nav_obs)
        return self.critic(features)

    def get_action_and_logprobs(self, observation: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        ss_obs, depth_obs, nav_obs = observation
        features = self._get_features(ss_obs, depth_obs, nav_obs)
        action_mean, action_log_std = torch.split(self.actor(features), self.action_dim, dim=1)
        
        dist = MultivariateNormal(action_mean, torch.diag_embed(torch.exp(action_log_std)))
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach(), log_prob.detach()
    
    def evaluate(self, observation, action):
        ss_obs, depth_obs, nav_obs = observation
        features = self._get_features(ss_obs, depth_obs, nav_obs)
        action_mean, action_log_std = torch.split(self.actor(features), self.action_dim, dim=1)
        
        dist = MultivariateNormal(action_mean, torch.diag_embed(torch.exp(action_log_std)))
        action = dist.sample()

        log_prob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        values = self.critic(features)

        return log_prob, values, dist_entropy
    
    def forward(self, observation: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        action, log_prob = self.get_action_and_logprobs(observation)
        value = self.get_value(observation)
        return action, value
