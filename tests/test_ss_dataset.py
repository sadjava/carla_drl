from torchvision import transforms
import torch
from unittest.mock import patch
from PIL import Image
import numpy as np

from carla_drl.semantic_segmentation.dataset import (
    CarlaSemanticSegmentationDataset,
    labels_to_cityscapes_palette
)
from carla_drl.semantic_segmentation import joint_transforms


def test_carla_semantic_segmentation_dataset():
    """Test the CarlaSemanticSegmentationDataset initialization."""
    joint_transform = joint_transforms.Compose([
        joint_transforms.Resize((480, 270)),
        joint_transforms.RandomHorizontalFlip()
    ])
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CarlaSemanticSegmentationDataset(root="data", joint_transform=joint_transform,
                                               input_transform=input_transform)
    sample = dataset[0]
    assert sample[0].shape == (3, 270, 480)
    assert sample[1].shape == (270, 480)
    assert sample[1].dtype == torch.int64
    assert sample[1].min() >= 0 and sample[1].max() < 13
    assert len(dataset) == 1

def test_random_horizontal_flip():
    """Test the RandomHorizontalFlip transformation."""
    transform = joint_transforms.RandomHorizontalFlip()
    img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3)).astype(np.uint8))
    mask = Image.fromarray(np.random.randint(0, 13, (100, 100)).astype(np.uint8))
    with patch('random.random', return_value=0.0):
        flipped_img, flipped_mask = transform(img, mask)
    assert not np.all(np.array(flipped_img) == np.array(img))
    assert not np.all(np.array(flipped_mask) == np.array(mask))
    with patch('random.random', return_value=1.0):
        not_flipped_img, not_flipped_mask = transform(img, mask)
    assert np.all(np.array(not_flipped_img) == np.array(img))
    assert np.all(np.array(not_flipped_mask) == np.array(mask))

def test_cityscapes_palette():
    """Test the cityscapes palette conversion."""
    label = torch.randint(0, 13, (270, 480)).long()
    palette = labels_to_cityscapes_palette(label)
    assert palette.size == (480, 270)
    assert type(palette) == Image.Image
