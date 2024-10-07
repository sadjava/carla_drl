from carla_drl.semantic_segmentation.dataset import (
    CarlaSemanticSegmentationDataset, 
    labels_to_cityscapes_palette
)
from torchvision import transforms
import torch


def test_carla_depth_estimation_dataset():
    """Test the CarlaDepthEstimationDataset initialization."""
    transform = transforms.Resize((360, 640))
    dataset = CarlaSemanticSegmentationDataset(root="data", transform=transform)
    sample = dataset[0]
    assert sample[0].shape == (3, 360, 640)
    assert sample[1].shape == (360, 640)
    assert sample[1].dtype == torch.int64
    assert sample[1].min() >= 0 and sample[1].max() < 13
    assert len(dataset) == 1

def test_cityscapes_palette():
    """Test the cityscapes palette conversion."""
    label = torch.randint(0, 13, (360, 640)).long()
    palette = labels_to_cityscapes_palette(label)
    assert palette.shape == (360, 640, 3)
    assert palette.dtype == torch.int16
    assert palette.min() >= 0 and palette.max() < 256
