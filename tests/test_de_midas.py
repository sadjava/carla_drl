from carla_drl.depth_estimation.midas import MonoDepthNet, resize_depth, resize_image
import torch
from unittest.mock import patch


def test_midas_depth_estimation_model():
    """Test the MonoDepthNet model."""
    model = MonoDepthNet()
    model.eval()
    with torch.no_grad():
        image = torch.randn(1, 3, 360, 640)
        x = resize_image(image)
        y = model(x)
        depth = resize_depth(y, 360, 640)
    assert depth.shape == (1, 1, 360, 640)
    with torch.no_grad():
        image = torch.randn(1, 3, 640, 360)
        x = resize_image(image)
        y = model(x)
        depth = resize_depth(y, 640, 360)
    assert depth.shape == (1, 1, 640, 360)

def test_load_midas_model():
    """Test the load method of the MonoDepthNet model."""
    model = MonoDepthNet()

    def mock_torch_load(path, *args, **kwargs):
        if path == "path/to/model.pth":
            return model.state_dict()
        else:
            # Use the original torch.load for other paths
            return original_torch_load(path, *args, **kwargs)

    # Save the original torch.load function
    original_torch_load = torch.load

    with patch.object(torch, 'load', side_effect=mock_torch_load):
        with patch.object(MonoDepthNet, 'load_state_dict',
                          wraps=model.load_state_dict) as mock_load_state_dict:
            MonoDepthNet("path/to/model.pth")
            assert mock_load_state_dict.call_count == 1
