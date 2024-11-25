import torch
from carla_drl.lane_following.model import ActorCritic


OBS_DIM = 100
ACTION_DIM = 2
STD_INIT = 0.1

def test_actor_critic_initialization():
    """Test the initialization of the ActorCritic class."""
    
    model = ActorCritic(OBS_DIM, ACTION_DIM, STD_INIT)
    
    assert model.action_dim == ACTION_DIM, "Action dimension is incorrect."
    assert len(model.actor) > 0, "Actor head is not initialized."
    assert len(model.critic) > 0, "Critic head is not initialized."

def test_actor_critic_forward():
    """Test the forward method of the ActorCritic class."""
    model = ActorCritic(OBS_DIM, ACTION_DIM, STD_INIT)

    observation = torch.randn(1, 100)
    
    action, value = model(observation)
    
    assert action.shape == (1, ACTION_DIM), "Action output shape is incorrect."
    assert value.shape == (1, 1), "Value output shape is incorrect."

def test_get_value():
    """Test the get_value method of the ActorCritic class."""

    model = ActorCritic(OBS_DIM, ACTION_DIM, STD_INIT)
 
    observation = torch.randn(1, 100)
    
    value = model.get_value(observation)
    
    assert value.shape == (1, 1), "Value output shape from get_value is incorrect."

def test_get_action_and_logprobs():
    """Test the get_action_and_logprobs method of the ActorCritic class."""

    model = ActorCritic(OBS_DIM, ACTION_DIM, STD_INIT)

    observation = torch.randn(1, 100)
    
    action, log_prob = model.get_action_and_log_prob(observation)
    
    assert action.shape == (1, ACTION_DIM), "Action output shape from get_action_and_logprobs is incorrect."
    assert log_prob.shape == (1,), "Log probability output shape is incorrect."

def test_evaluate():
    """Test the evaluate method of the ActorCritic class."""

    model = ActorCritic(OBS_DIM, ACTION_DIM, STD_INIT)

    observation = torch.randn(1, 100)
    
    action = torch.randn(1, ACTION_DIM)  # Mock action
    log_prob, values, dist_entropy = model.evaluate(observation, action)
    
    assert log_prob.shape == (1,), "Log probability output shape from evaluate is incorrect."
    assert values.shape == (1, 1), "Value output shape from evaluate is incorrect."
    assert dist_entropy.shape == (1,), "Entropy output shape from evaluate is incorrect."
