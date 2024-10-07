from carla_drl.semantic_segmentation.unet import UNet
import torch


def test_unet_model():
    """Test the UNet model."""
    model = UNet(num_classes=13)
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 360, 640)
        y = model(x)
    assert y.shape == (1, 13, 360, 640)
