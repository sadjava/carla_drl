from carla_drl.depth_estimation.midas import MonoDepthNet, resize_depth, resize_image
import torch
from unittest.mock import patch


def test_midas_depth_estimation_model():
    """Test the MonoDepthNet model."""
    model = MonoDepthNet()
    model.eval()
    with torch.no_grad():
        image = torch.randn(1, 3, 90, 160)
        N, _, H, W = image.shape
        x = resize_image(image)
        y = model(x)
        depth = resize_depth(y, H, W)
    assert depth.shape == (N, 1, H, W)
    with torch.no_grad():
        image = torch.randn(1, 3, 160, 90)
        N, _, H, W = image.shape
        x = resize_image(image)
        y = model(x)
        depth = resize_depth(y, H, W)
    assert depth.shape == (N, 1, H, W)
    assert torch.all(torch.logical_and(depth >= 0, depth <= 1))

def test_get_features():
    """Test the get_features method of the MonoDepthNet model."""
    model = MonoDepthNet()
    model.eval()
    with torch.no_grad():
        image = torch.randn(1, 3, 90, 160)
        N, _, H, W = image.shape
        x = resize_image(image)
        y = model.get_features(x)
    assert y.shape == (N, 128, 14, 24)

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
