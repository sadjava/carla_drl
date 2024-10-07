from carla_drl.depth_estimation.dataset import CarlaDepthEstimationDataset
from torchvision import transforms


def test_carla_depth_estimation_dataset():
    """Test the CarlaDepthEstimationDataset initialization."""
    transform = transforms.Resize((360, 640))
    dataset = CarlaDepthEstimationDataset(root="data", transform=transform)
    sample = dataset[0]
    assert sample[0].shape == (3, 360, 640)
    assert sample[1].shape == (360, 640)
    assert sample[1].min() >= 0.0 and sample[1].max() <= 1.0
    assert len(dataset) == 1