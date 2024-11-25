import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, action_std_init, device: str = 'cpu'):
        super(ActorCritic, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        
        self.cov_var = torch.full((self.action_dim,), action_std_init)

        self.cov_mat = torch.diag(self.cov_var).unsqueeze(dim=0)

        self.actor = nn.Sequential(
                        nn.Linear(obs_dim, 500),
                        nn.Tanh(),
                        nn.Linear(500, 300),
                        nn.Tanh(),
                        nn.Linear(300, 100),
                        nn.Tanh(),
                        nn.Linear(100, self.action_dim),
                        nn.Tanh()
                    )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(obs_dim, 500),
                        nn.Tanh(),
                        nn.Linear(500, 300),
                        nn.Tanh(),
                        nn.Linear(300, 100),
                        nn.Tanh(),
                        nn.Linear(100, 1)
                    )

    def forward(self, obs):
        return self.actor(obs), self.critic(obs)
    
    def set_action_std(self, new_action_std):
        self.cov_var = torch.full((self.action_dim,), new_action_std)


    def get_value(self, obs):
        obs = torch.tensor(obs, dtype=torch.float)
        return self.critic(obs)
    
    def get_action_and_log_prob(self, obs):
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.detach(), log_prob.detach()
    
    def evaluate(self, obs, action):

        mean = self.actor(obs)
        cov_var = self.cov_var.expand_as(mean)
        cov_mat = torch.diag_embed(cov_var)
        dist = MultivariateNormal(mean, cov_mat)
        
        logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        values = self.critic(obs)
        
        return logprobs, values, dist_entropy