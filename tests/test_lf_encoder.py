import torch
import numpy as np
from unittest.mock import patch

from carla_drl.lane_following.encoder import ObservationEncoder
from carla_drl.depth_estimation.midas import MonoDepthNet
from carla_drl.semantic_segmentation.unet import UNet

def test_observation_encoder():
    """Test the ObservationEncoder class."""
    
    # Mock paths for the models
    ss_model_path = "path/to/ss_model.pth"
    depth_model_path = "path/to/depth_model.pth"
    image_shape = (270, 480)
    device = 'cpu'
    
    ss_model = UNet(num_classes=13)
    depth_model = MonoDepthNet()

    def mock_torch_load(path, *args, **kwargs):
        if path == "path/to/ss_model.pth":
            return ss_model.state_dict()
        elif path == "path/to/depth_model.pth":
            return depth_model.state_dict()
        else:
            # Use the original torch.load for other paths
            return original_torch_load(path, *args, **kwargs)

    # Save the original torch.load function
    original_torch_load = torch.load

    with patch.object(torch, 'load', side_effect=mock_torch_load):
        with patch.object(UNet, 'load_state_dict', wraps=ss_model.load_state_dict) as mock_ss_load_state_dict:
            with patch.object(MonoDepthNet, 'load_state_dict', wraps=depth_model.load_state_dict) as mock_depth_load_state_dict:
                # Create an instance of the ObservationEncoder
                encoder = ObservationEncoder(ss_model_path, depth_model_path, image_shape, device)
    
    # Create a mock observation
    image_obs = np.random.randint(0, 255, (270, 480, 3), dtype=np.uint8)
    nav_obs = np.random.rand(5)
    observation = (image_obs, nav_obs)
    
    # Process the observation
    ss_obs, depth_obs, nav_obs = encoder.process(observation)
    
    # Check the shapes of the outputs
    assert ss_obs.shape == (1, 128, 16, 30), "Semantic segmentation output shape is incorrect."
    assert depth_obs.shape == (1, 128, 7, 12), "Depth output shape is incorrect."
    assert nav_obs.shape == (1, 5), "Navigation output shape is incorrect."
    
    # Check the device of the outputs
    assert ss_obs.device == torch.device(device), "Semantic segmentation output device is incorrect."
    assert depth_obs.device == torch.device(device), "Depth output device is incorrect."
    assert nav_obs.device == torch.device(device), "Navigation output device is incorrect."
