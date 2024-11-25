from carla_drl.autoencoder.model import VariationalAutoencoder
import torch
from unittest.mock import patch


def test_vae():
    """Test the MonoDepthNet model."""
    model = VariationalAutoencoder(3, 95)
    model.eval()
    with torch.no_grad():
        image = torch.randn(1, 3, 80, 160)
        y = model.encoder(image)
        reconstructed = model.decoder(y)
    assert reconstructed.shape == image.shape

def test_forwrad():
    """Test the MonoDepthNet model."""
    model = VariationalAutoencoder(3, 95)
    model.eval()
    with torch.no_grad():
        image = torch.randn(1, 3, 80, 160)
        N, _, H, W = image.shape
        y = model(image)
    assert y.shape == (N, 95)