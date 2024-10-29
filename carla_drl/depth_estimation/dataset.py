import numpy as np
import glob
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple


def rgb_to_depth(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to a depth image.
    """
    normalized_depth = np.dot(image[:, :, :3][:, :, ::-1], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0
    return normalized_depth

class CarlaDepthEstimationDataset(Dataset):
    """
    Dataset for Carla depth estimation.
    """
    def __init__(self, root: str, joint_transform: transforms.Compose = None, input_transform: transforms.Compose = None):
        self.root = root
        self.joint_transform = joint_transform
        self.input_transform = input_transform
        self.images = sorted(glob.glob(os.path.join(root, "rgb", "*", "*", "*.png")))
        self.labels = sorted(glob.glob(os.path.join(root, "depth", "*", "*", "*.png")))
        for image, label in zip(self.images, self.labels):
            assert image.split("/")[-1] == label.split("/")[-1], "Image and label do not match"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[idx]).convert("RGB")
        depth  = Image.open(self.labels[idx])
        if self.joint_transform:
            image, depth = self.joint_transform(image, depth)
        if self.input_transform:
            image = self.input_transform(image)
        depth = rgb_to_depth(np.array(depth))
        depth = torch.from_numpy(depth).unsqueeze(0)
        return image, depth
