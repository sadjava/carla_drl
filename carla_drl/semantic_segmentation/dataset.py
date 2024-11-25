import numpy as np
import glob
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from typing import Tuple


def cityscapes_palette_to_labels(image: np.ndarray) -> np.ndarray:
    """
    Convert an image containing CARLA semantic segmentation labels to
    Cityscapes palette.
    """
    classes = {
        0: (0, 0, 0),            # Unlabeled
        1: (128, 64, 128),       # Roads
        2: (244, 35, 232),       # SideWalks
        3: (70, 70, 70),         # Building
        4: (102, 102, 156),      # Wall
        5: (190, 153, 153),      # Fence
        6: (153, 153, 153),      # Pole
        7: (107, 142, 35),       # Vegetation
        8: (152, 251, 152),      # Terrain
        9: (70, 130, 180),       # Sky
        10: (45, 60, 150),       # Water
        11: (157, 234, 50),      # RoadLine
        12: (81, 0, 81),         # Ground
    }
    array = image[..., :3]
    result = np.zeros((array.shape[0], array.shape[1]))
    for key, value in classes.items():
        result[np.where(np.all(array == value, axis=-1))] = key
    return result

def labels_to_cityscapes_palette(image: np.ndarray) -> Image.Image:
    """
    Convert an image containing CARLA semantic segmentation labels to
    Cityscapes palette.
    """
    classes = {
        0: (0, 0, 0),            # Unlabeled
        1: (128, 64, 128),       # Roads
        2: (244, 35, 232),       # SideWalks
        3: (70, 70, 70),         # Building
        4: (102, 102, 156),      # Wall
        5: (190, 153, 153),      # Fence
        6: (153, 153, 153),      # Pole
        7: (107, 142, 35),       # Vegetation
        8: (152, 251, 152),      # Terrain
        9: (70, 130, 180),       # Sky
        10: (45, 60, 150),       # Water
        11: (157, 234, 50),      # RoadLine
        12: (81, 0, 81),         # Ground
    }
    result = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for key, value in classes.items():
        result[np.where(image == key)] = value
    return Image.fromarray(result)

class CarlaSemanticSegmentationDataset(Dataset):
    """
    Dataset for Carla semantic segmentation.
    """
    NUM_CLASSES = 13
    def __init__(self, root: str, joint_transform: transforms.Compose = None, input_transform: transforms.Compose = None):
        self.root = root
        self.joint_transform = joint_transform
        self.input_transform = input_transform
        self.images = sorted(glob.glob(os.path.join(root, "rgb", "*", "*", "*.png")))
        self.labels = sorted(glob.glob(os.path.join(root, "semseg", "*", "*", "*.png")))
        for image, label in zip(self.images, self.labels):
            assert image.split("/")[-1] == label.split("/")[-1], "Image and label do not match"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[idx]).convert("RGB")
        label = Image.open(self.labels[idx])
        if self.joint_transform:
            image, label = self.joint_transform(image, label)
        if self.input_transform:
            image = self.input_transform(image)
        label = cityscapes_palette_to_labels(np.array(label))
        label = torch.from_numpy(label).long()
        return image, label
