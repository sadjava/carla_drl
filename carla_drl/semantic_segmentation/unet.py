import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: bool = False):
        super(_EncoderBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the encoder block."""
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, middle_channels: int, out_channels: int, output_padding: Tuple[int, int] = (0, 0)):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                middle_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                output_padding=output_padding
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the decoder block."""
        return self.decode(x)


class UNet(nn.Module):
    """UNet model for semantic segmentation."""
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(3, 16)
        self.enc2 = _EncoderBlock(16, 32)
        self.enc3 = _EncoderBlock(32, 64)
        self.enc4 = _EncoderBlock(64, 128, dropout=True)
        self.center = _DecoderBlock(128, 256, 128, output_padding=(1, 0))
        self.dec4 = _DecoderBlock(256, 128, 64, output_padding=(1, 0))
        self.dec3 = _DecoderBlock(128, 64, 32, output_padding=(1, 0))
        self.dec2 = _DecoderBlock(64, 32, 16)
        self.dec1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize the weights of the model."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the UNet model."""
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        dec4 = self.dec4(torch.cat([
            center,
            F.interpolate(enc4, center.size()[2:], mode='bilinear')
        ], 1))
        dec3 = self.dec3(torch.cat([
            dec4,
            F.interpolate(enc3, dec4.size()[2:], mode='bilinear')
        ], 1))
        dec2 = self.dec2(torch.cat([
            dec3,
            F.interpolate(enc2, dec3.size()[2:], mode='bilinear')
        ], 1))
        dec1 = self.dec1(torch.cat([
            dec2,
            F.interpolate(enc1, dec2.size()[2:], mode='bilinear')
        ], 1))
        final = self.final(dec1)
        return F.interpolate(final, x.size()[2:], mode='bilinear')

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Feature extraction of the UNet model."""
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)

        return center
    
    def load(self, path: str):
        """Load model from file.

        Args:
            path (str): file path
        """

        self.load_state_dict(torch.load(path, map_location='cpu'))
