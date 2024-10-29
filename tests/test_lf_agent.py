import torch
import numpy as np
from unittest.mock import patch

from carla_drl.lane_following.agent import PPOAgent

def test_ppo_agent_initialization():
    """Test the initialization of the PPOAgent class."""
    agent = PPOAgent()
    
    assert agent.ss_channels == 128, "Semantic segmentation channels are incorrect."
    assert agent.depth_channels == 128, "Depth channels are incorrect."
    assert agent.action_dim == 2, "Action dimension is incorrect."
    assert agent.batch_size == 64, "Batch size is incorrect."
    assert agent.lr == 1e-4, "Learning rate is incorrect."
    assert agent.gamma == 0.99, "Gamma is incorrect."
    assert agent.epsilon == 0.2, "Epsilon is incorrect."
    assert agent.epoch_n == 30, "Number of epochs is incorrect."
    assert agent.device == 'cpu', "Device is incorrect."
    assert len(agent.memory.ss_obs) == 0, "Memory should be empty upon initialization."

def test_get_action():
    """Test the get_action method of the PPOAgent class."""
    agent = PPOAgent()
    
    ss_obs = torch.randn(1, 128, 16, 30)
    depth_obs = torch.randn(1, 128, 7, 12)
    nav_obs = torch.randn(1, 5)
    observation = (ss_obs, depth_obs, nav_obs)
    
    action = agent.get_action(observation)
    
    assert action.shape == (2,), "Action output shape is incorrect."
    assert action.dtype == np.float32, "Action output type is incorrect."

def test_learn():
    """Test the learn method of the PPOAgent class."""
    agent = PPOAgent()
    
    # Simulate adding data to the memory
    for _ in range(8):
        agent.memory.ss_obs.append(torch.randn(1, 128, 16, 30))
        agent.memory.depth_obs.append(torch.randn(1, 128, 7, 12))
        agent.memory.nav_obs.append(torch.randn(1, 5))
        agent.memory.actions.append(torch.randn(1, 2))
        agent.memory.log_probs.append(torch.randn(1,))
        agent.memory.rewards.append(1.0)
        agent.memory.dones.append(True)

    # Call the learn method
    agent.learn()
    
    assert len(agent.memory.ss_obs) == 0, "Memory should be cleared after learning."

def test_save_load():
    """Test the save and load methods of the PPOAgent class."""
    agent = PPOAgent()
    path = "test_agent.pth"
    
    # Save the model
    agent.save(path)
    
    # Create a new agent and load the model
    new_agent = PPOAgent()
    new_agent.load(path)
    
    # Check if the loaded model parameters are the same
    for param1, param2 in zip(agent.policy.parameters(), new_agent.policy.parameters()):
        assert torch.equal(param1, param2), "Loaded model parameters do not match."

