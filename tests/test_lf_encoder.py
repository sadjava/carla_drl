import torch
import numpy as np
from unittest.mock import patch

from carla_drl.lane_following.encoder import ObservationEncoder
from carla_drl.autoencoder.model import VariationalAutoencoder
from carla_drl.semantic_segmentation.unet import UNet

def test_observation_encoder():
    """Test the ObservationEncoder class."""
    
    # Mock paths for the models
    ss_model_path = "path/to/ss_model.pth"
    ae_model_path = "path/to/depth_model.pth"
    image_shape = (80, 160)
    latent_dims = 95
    device = 'cpu'
    
    ss_model = UNet(num_classes=13)
    ae_model = VariationalAutoencoder(latent_dims=95)

    def mock_torch_load(path, *args, **kwargs):
        if path == "path/to/ss_model.pth":
            return ss_model.state_dict()
        elif path == "path/to/depth_model.pth":
            return ae_model.state_dict()
        else:
            # Use the original torch.load for other paths
            return original_torch_load(path, *args, **kwargs)

    # Save the original torch.load function
    original_torch_load = torch.load

    with patch.object(torch, 'load', side_effect=mock_torch_load):
        with patch.object(UNet, 'load_state_dict', wraps=ss_model.load_state_dict) as mock_ss_load_state_dict:
            with patch.object(VariationalAutoencoder, 'load_state_dict', wraps=ae_model.load_state_dict) as mock_depth_load_state_dict:
                # Create an instance of the ObservationEncoder
                encoder = ObservationEncoder(ss_model_path, ae_model_path, image_shape, latent_dims, device)
    
    # Create a mock observation
    image_obs = np.random.randint(0, 255, (80, 160, 3), dtype=np.uint8)
    nav_obs = np.random.rand(5)
    observation = (image_obs, nav_obs)
    
    # Process the observation
    obs = encoder.process(observation)
    
    # Check the shapes of the outputs
    assert obs.shape == (1, 100), "Observation output shape is incorrect."
    
    # Check the device of the outputs
    assert obs.device == torch.device(device), "Navigation output device is incorrect."
