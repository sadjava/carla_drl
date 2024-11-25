import torch
import numpy as np

from carla_drl.lane_following.agent import PPOAgent

def test_get_action():
    """Test the get_action method of the PPOAgent class."""
    agent = PPOAgent("Town02")
    
    obs = torch.randn(1, 55)
    
    action = agent.get_action(obs)
    
    assert action.shape == (2,), "Action output shape is incorrect."
    assert action.dtype == np.float32, "Action output type is incorrect."

def test_learn():
    """Test the learn method of the PPOAgent class."""
    agent = PPOAgent("Town02")
    
    # Simulate adding data to the memory
    for _ in range(8):
        agent.memory.states.append(torch.randn(1, 55))
        agent.memory.actions.append(torch.randn(1, 2))
        agent.memory.log_probs.append(torch.randn(1,))
        agent.memory.rewards.append(1.0)
        agent.memory.dones.append(True)

    # Call the learn method
    agent.learn()
    
    assert len(agent.memory.states) == 0, "Memory should be cleared after learning."

def test_save_load():
    """Test the save and load methods of the PPOAgent class."""
    agent = PPOAgent("Town02")
    path = "test_agent.pth"
    
    # Save the model
    agent.save(path)
    
    # Create a new agent and load the model
    new_agent = PPOAgent("Town02")
    new_agent.load(path)
    
    # Check if the loaded model parameters are the same
    for param1, param2 in zip(agent.policy.parameters(), new_agent.policy.parameters()):
        assert torch.equal(param1, param2), "Loaded model parameters do not match."

def test_action_std():
    agent = PPOAgent("Town02")
    
    agent.set_action_std(0.2)
    assert agent.action_std == 0.2, "Action std set incorrectly"

    agent.decay_action_std(0.05, 0.05)
    assert np.isclose(agent.action_std, 0.15), "Action std decayed incorrectly"


    agent.decay_action_std(0.05, 0.12)
    assert np.isclose(agent.action_std, 0.12), "Action std decayed incorrectly"