import torch
import torch.nn as nn


class VariationalEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dims: int = 50):  
        super(VariationalEncoder, self).__init__()

        self.encoder_layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2),
            nn.LeakyReLU())

        self.encoder_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())

        self.encoder_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2),
            nn.LeakyReLU())

        self.encoder_layer4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU())

        self.linear = nn.Sequential(
            nn.Linear(9*4*256, 1024),
            nn.LeakyReLU())

        self.mu = nn.Linear(1024, latent_dims)
        self.sigma = nn.Linear(1024, latent_dims)

        self.kl = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        x = self.encoder_layer4(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        mu =  self.mu(x)
        sigma = torch.exp(self.sigma(x))
        eps = torch.randn_like(sigma)
        z = mu + sigma*eps
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

class Decoder(nn.Module):
    
    def __init__(self, in_channels: int = 3, latent_dims: int = 50):
        super().__init__()
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dims, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 9 * 4 * 256),
            nn.LeakyReLU()
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256,4,9))

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 4,  stride=2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2,
                               padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2),
            nn.Sigmoid())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder_linear(x)
        x = self.unflatten(x)
        x = self.decoder(x)
        return x


class VariationalAutoencoder(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dims: int = 50):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(in_channels, latent_dims)
        self.decoder = Decoder(in_channels, latent_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return z
    
    def load(self, path: str):
        self.load_state_dict(torch.load(path))
