from torchvision import transforms
from PIL import Image
import numpy as np
from unittest.mock import patch

from carla_drl.depth_estimation.dataset import CarlaDepthEstimationDataset
from carla_drl.depth_estimation import joint_transforms


def test_carla_depth_estimation_dataset():
    """Test the CarlaDepthEstimationDataset initialization."""
    joint_transform = joint_transforms.Compose([
        joint_transforms.Resize((480, 270)),
        joint_transforms.RandomHorizontalFlip()
    ])
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = CarlaDepthEstimationDataset(root="data", joint_transform=joint_transform, input_transform=input_transform)
    sample = dataset[0]
    assert sample[0].shape == (3, 270, 480)
    assert sample[1].shape == (1, 270, 480)
    assert sample[1].min() >= 0.0 and sample[1].max() <= 1.0
    assert len(dataset) == 1

def test_random_horizontal_flip():
    """Test the RandomHorizontalFlip transformation."""
    transform = joint_transforms.RandomHorizontalFlip()
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8))
    mask = Image.fromarray(np.random.randint(0, 29, (100, 100)).astype(np.uint8))
    with patch('random.random', return_value=0.0):
        flipped_img, flipped_mask = transform(img, mask)
    assert not np.all(np.array(flipped_img) == np.array(img))
    assert not np.all(np.array(flipped_mask) == np.array(mask))
    with patch('random.random', return_value=1.0):
        not_flipped_img, not_flipped_mask = transform(img, mask)
    assert np.all(np.array(not_flipped_img) == np.array(img))
    assert np.all(np.array(not_flipped_mask) == np.array(mask))
