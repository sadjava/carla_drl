import torch
from torchvision import transforms
from typing import Tuple
import numpy as np
from PIL import Image

from carla_drl.semantic_segmentation.unet import UNet
from carla_drl.depth_estimation.midas import MonoDepthNet, resize_image

class ObservationEncoder:
    def __init__(self, ss_model_path: str, depth_model_path: str, image_shape: Tuple[int], device: str = 'cpu'):
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize(image_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        self.ss_model = UNet(num_classes=13)
        for p in self.ss_model.parameters():
            p.requires_grad = False
        self.ss_model.load(ss_model_path)
        self.ss_model.to(device)
        
        self.depth_model = MonoDepthNet()
        for p in self.depth_model.parameters():
            p.requires_grad = False
        self.depth_model.load(depth_model_path)
        self.depth_model.to(device)

        self.device = device

    def process(self, observation: Tuple[np.ndarray, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_obs, nav_obs = observation
        image_obs = Image.fromarray(image_obs)
        image_obs = self.transform(image_obs).unsqueeze(0).to(self.device)

        ss_obs = self.ss_model.get_features(image_obs)
        depth_obs = self.depth_model.get_features(resize_image(image_obs))
        nav_obs = torch.tensor(nav_obs, dtype=torch.float, device=self.device).unsqueeze(0)

        return (ss_obs, depth_obs, nav_obs)
