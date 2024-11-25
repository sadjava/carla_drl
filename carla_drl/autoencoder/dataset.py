import glob
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple
import numpy as np


def rgb_to_depth(image: np.ndarray) -> torch.Tensor:
    """
    Convert an RGB image to a depth image.
    """
    normalized_depth = np.dot(image[:, :, :3][:, :, ::-1], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0
    return torch.from_numpy(normalized_depth).unsqueeze(0)


class CarlaDataset(Dataset):
    """
    Dataset for Carla depth estimation or semantic segmentation.
    """
    def __init__(self, root: str, is_depth: bool, transform: transforms.Compose = None):
        self.mode = "depth" if is_depth else "semseg"
        self.root = os.path.join(root, self.mode)
        self.is_depth = is_depth
        self.transform = transform
        self.images = sorted(glob.glob(os.path.join(self.root, "*", "*", "*.png")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.is_depth:
            image = rgb_to_depth(np.array(image.permute(1, 2, 0))).float()
        return image, 0
