"""MonoDepthNet: Network for monocular depth estimation trained by mixing several datasets.
This file contains code that is adapted from
https://github.com/thomasjpfan/pytorch_refinenet/blob/master/pytorch_refinenet/refinenet/refinenet_4cascade.py
"""
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class MonoDepthNet(nn.Module):
    """Network for monocular depth estimation.
    """

    def __init__(self, path: str = None, features: int = 128):
        """Init.

        Args:
            path (str, optional): Path to saved model. Defaults to None.
            features (int, optional): Number of features. Defaults to 256.
        """
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.pretrained = nn.Module()
        self.scratch = nn.Module()
        self.pretrained.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                               resnet.maxpool, resnet.layer1)

        self.pretrained.layer2 = resnet.layer2
        self.pretrained.layer3 = resnet.layer3
        self.pretrained.layer4 = resnet.layer4

        # adjust channel number of feature maps
        self.scratch.layer1_rn = nn.Conv2d(
            64, features, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.scratch.layer2_rn = nn.Conv2d(
            128, features, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.scratch.layer3_rn = nn.Conv2d(
            256, features, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.scratch.layer4_rn = nn.Conv2d(
            512, features, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.scratch.refinenet4 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet1 = FeatureFusionBlock(features)

        # adaptive output module: 2 convolutions and upsampling
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode='bilinear')
        )

        # load model
        if path:
            self.load(path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)
        return torch.sigmoid(out)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (tensor): input data (image)

        Returns:
            tensor: depth
        """
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_4_rn = self.scratch.layer4_rn(layer_4)
        path_4 = self.scratch.refinenet4(layer_4_rn)
        return path_4

    def load(self, path: str):
        """Load model from file.

        Args:
            path (str): file path
        """

        self.load_state_dict(torch.load(path, map_location='cpu'))


class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor: float, mode: str):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)

        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features: int):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features: int):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.res_conf_unit = ResidualConvUnit(features)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.res_conf_unit(xs[1])

        output = self.res_conf_unit(output)
        output = nn.functional.interpolate(output, scale_factor=2,
                                           mode='bilinear', align_corners=True)

        return output

def resize_image(img: torch.Tensor) -> torch.Tensor:
    """Resize image and make it fit for network.

    Args:
        img (tensor): input image tensor of shape BxCxHxW

    Returns:
        tensor: resized image tensor
    """
    B, C, H, W = img.shape

    if W > H:
        scale = W / 384
    else:
        scale = H / 384

    new_H = int(np.ceil(H / scale / 32) * 32)
    new_W = int(np.ceil(W / scale / 32) * 32)

    # Use torch.nn.functional.interpolate for resizing
    img_resized = torch.nn.functional.interpolate(
        img,
        size=(new_H, new_W),
        mode='bilinear',
        align_corners=False
    )

    return img_resized

def resize_depth(depth: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize depth map tensor.

    Args:
        depth (tensor): depth tensor of shape Bx1xHxW
        width (int): target width
        height (int): target height

    Returns:
        tensor: resized depth tensor of shape Bx1xHxW
    """
    # Use torch.nn.functional.interpolate for resizing
    depth_resized = torch.nn.functional.interpolate(
        depth,
        size=(height, width),
        mode='bilinear',
        align_corners=False
    )

    return depth_resized
