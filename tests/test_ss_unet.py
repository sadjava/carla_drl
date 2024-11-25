import torch
from unittest.mock import patch

from carla_drl.semantic_segmentation.unet import UNet


def test_unet_model():
    """Test the UNet model."""
    model = UNet(num_classes=13)
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 80, 160)
        y = model(x)
    assert y.shape == (1, 13, 80, 160)

def test_get_features():
    """Test the get_features method of the UNet model."""
    model = UNet(num_classes=13)
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 80, 160)
        y = model.get_features(x)
    print(y.shape)
    assert y.shape == (1, 128, 11, 20)

def test_load_unet_model():
    """Test the load method of the UNet model."""
    model = UNet(num_classes=13)

    def mock_torch_load(path, *args, **kwargs):
        if path == "path/to/model.pth":
            return model.state_dict()
        else:
            # Use the original torch.load for other paths
            return original_torch_load(path, *args, **kwargs)

    # Save the original torch.load function
    original_torch_load = torch.load

    with patch.object(torch, 'load', side_effect=mock_torch_load):
        with patch.object(UNet, 'load_state_dict',
                          wraps=model.load_state_dict) as mock_load_state_dict:
            model.load("path/to/model.pth")
            assert mock_load_state_dict.call_count == 1
