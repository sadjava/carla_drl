import numpy as np
import glob
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

def cityscapes_palette_to_labels(image):
    """
    Convert an image containing CARLA semantic segmentation labels to
    Cityscapes palette.
    """
    classes = {
        0: [0, 0, 0],         # None
        1: [70, 70, 70],      # Buildings
        2: [190, 153, 153],   # Fences
        3: [72, 0, 90],       # Other
        4: [220, 20, 60],     # Pedestrians
        5: [153, 153, 153],   # Poles
        6: [157, 234, 50],    # RoadLines
        7: [128, 64, 128],    # Roads
        8: [244, 35, 232],    # Sidewalks
        9: [107, 142, 35],    # Vegetation
        10: [0, 0, 255],      # Vehicles
        11: [102, 102, 156],  # Walls
        12: [220, 220, 0]     # TrafficSigns
    }
    array = image[..., :3]
    result = np.zeros((array.shape[0], array.shape[1]))
    for key, value in classes.items():
        result[np.where(np.all(array == value, axis=-1))] = key
    return result

def labels_to_cityscapes_palette(image):
    """
    Convert an image containing CARLA semantic segmentation labels to
    Cityscapes palette.
    """
    classes = {
        0: [0, 0, 0],         # None
        1: [70, 70, 70],      # Buildings
        2: [190, 153, 153],   # Fences
        3: [72, 0, 90],       # Other
        4: [220, 20, 60],     # Pedestrians
        5: [153, 153, 153],   # Poles
        6: [157, 234, 50],    # RoadLines
        7: [128, 64, 128],    # Roads
        8: [244, 35, 232],    # Sidewalks
        9: [107, 142, 35],    # Vegetation
        10: [0, 0, 255],      # Vehicles
        11: [102, 102, 156],  # Walls
        12: [220, 220, 0]     # TrafficSigns
    }
    result = torch.zeros((image.shape[0], image.shape[1], 3), dtype=torch.int16)
    for key, value in classes.items():
        result[np.where(image == key)] = torch.tensor(value, dtype=torch.int16)
    return result

class CarlaSemanticSegmentationDataset(Dataset):
    """
    Dataset for Carla semantic segmentation.
    """
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = sorted(glob.glob(os.path.join(root, "rgb", "*.png")))
        self.labels = sorted(glob.glob(os.path.join(root, "semseg", "*.png")))
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
        label = Image.open(self.labels[idx])
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        label = cityscapes_palette_to_labels(np.array(label))
        image = self.post_transform(image)
        label = torch.from_numpy(label).long()
        return image, label
