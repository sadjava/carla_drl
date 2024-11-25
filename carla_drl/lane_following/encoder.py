import torch
from torchvision import transforms
import torchvision.transforms.functional as F 
from typing import Tuple
import numpy as np
from PIL import Image
import pygame
import time

from carla_drl.semantic_segmentation.unet import UNet
from carla_drl.autoencoder.model import VariationalAutoencoder

class ObservationEncoder:
    def __init__(self, ss_model_path: str, ss_ae_model_path: str, de_ae_model_path: str = None, image_shape: Tuple[int] = (80, 160), latent_dims: int = 50, use_depth: bool = False, device: str = 'cpu'):
        
        # self.display = pygame.display.set_mode((320, 160 * (2 + use_depth)),pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.device = device
        self.use_depth = use_depth

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize(image_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.ss_model = UNet(num_classes=13)
        self.ss_ae = VariationalAutoencoder(3, latent_dims)
        self.ss_model.load(ss_model_path)
        for p in self.ss_model.parameters():
            p.requires_grad = False
        self.ss_model.to(device)
        self.ss_model.eval()
        
        self.ss_ae.load(ss_ae_model_path)
        for p in self.ss_ae.parameters():
            p.requires_grad = False
        self.ss_ae.to(device)

        if self.use_depth:
            self.depth_ae = VariationalAutoencoder(1, latent_dims)
            self.depth_ae.load(de_ae_model_path)
            for p in self.depth_ae.parameters():
                p.requires_grad = False
            self.depth_ae.to(device)

        self.record = False
        self.images = []
        self.start_time = time.time()

    def process(self, observation: Tuple[np.ndarray, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_depth:
            orig_img, depth_obs, nav_obs = observation
        else:
            orig_img, nav_obs = observation

        im = Image.fromarray(orig_img).convert('RGB')
        image_obs = self.transform(im).unsqueeze(0).to(self.device)
        
        if self.use_depth:
            depth_im = Image.fromarray(255 * self.rgb_to_depth(depth_obs))
        
        ss_outputs = self.ss_model(image_obs)
        ss_prediction = ss_outputs.data.max(1)[1].squeeze_(1).cpu().numpy()
        ss_im = self.labels_to_cityscapes_palette(ss_prediction[0])

        # depth_outputs = resize_depth(self.depth_model(resize_image(image_obs)), orig_img.shape[0], orig_img.shape[1])
        # depth_prediction = depth_outputs[0][0].cpu().numpy()
        # depth_im = Image.fromarray((depth_prediction * 255).astype(np.uint8))

        # if self.use_depth:
        #     concat_image = np.concatenate((
        #         np.array(im.resize((320, 160))),
        #         np.array(ss_im.resize((320, 160))),
        #         np.array(depth_im.resize((320, 160)).convert("RGB")),
        #     ), axis=0)
        # else:
        #     concat_image = np.concatenate((
        #         np.array(im.resize((320, 160))),
        #         np.array(ss_im.resize((320, 160))),
        #     ), axis=0)
            
        # t = time.time()
        # if len(self.images) == 0 and t - self.start_time > 40:
        #     self.start_time = t
        #     self.record = True
        # if self.record:
        #     if len(self.images) == 0:
        #         self.start_time = time.time()
        #     self.images.append(Image.fromarray(concat_image))
        #     duration = time.time() - self.start_time
        #     if duration > 30:
        #         self.record = False
        #         self.images[0].save('info/gifs/front_view.gif', save_all=True, duration=duration, append_images=self.images[1:], optimize=False, loop=0)
            
        # self.surface = pygame.surfarray.make_surface(concat_image.swapaxes(0, 1))
        # self.display.blit(self.surface, (0, 0))
        # pygame.display.flip()

        ss_obs = self.ss_ae(F.to_tensor(ss_im).unsqueeze(0).to(self.device))
        if self.use_depth:
            depth_obs = self.depth_ae(F.to_tensor(depth_im).unsqueeze(0).to(self.device))
        nav_obs = torch.tensor(nav_obs, dtype=torch.float, device=self.device).unsqueeze(0)
        if self.use_depth:
            depth_obs = self.depth_ae(F.to_tensor(depth_im).unsqueeze(0).to(self.device))
            obs = torch.cat((ss_obs, depth_obs, nav_obs), dim=1)
        else:
            obs = torch.cat((ss_obs, nav_obs), dim=1)

        return obs
    
    def labels_to_cityscapes_palette(self, image: np.ndarray) -> Image.Image:
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
    
    def rgb_to_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Convert an RGB image to a depth image.
        """
        normalized_depth = np.dot(image[:, :, :3][:, :, ::-1], [65536.0, 256.0, 1.0])
        normalized_depth /= 16777215.0
        return normalized_depth
