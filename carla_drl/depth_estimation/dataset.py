import numpy as np
import glob
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def rgb_to_depth(image):
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
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = sorted(glob.glob(os.path.join(root, "rgb", "*.png")))
        self.labels = sorted(glob.glob(os.path.join(root, "depth", "*.png")))
        for image, label in zip(self.images, self.labels):
            assert image.split("/")[-1] == label.split("/")[-1], "Image and label do not match"
        self.post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        depth = Image.open(self.labels[idx])
        if self.transform:
            image = self.transform(image)
            depth = self.transform(depth)
        depth = rgb_to_depth(np.array(depth))
        image = self.post_transform(image)
        depth = torch.from_numpy(depth)
        return image, depth
