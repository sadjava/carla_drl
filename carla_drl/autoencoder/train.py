import os
import argparse
import datetime
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from carla_drl.autoencoder.dataset import CarlaDataset
from carla_drl.autoencoder.model import VariationalAutoencoder


def train(model, trainloader, optimizer, device):
    model.train()
    train_loss = 0.0
    for x, _ in trainloader:
        x = x.to(device)
        x_hat = model.decoder(model.encoder(x))
        loss = ((x - x_hat) ** 2).sum() + model.encoder.kl
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() / x.shape[0]
    return train_loss / len(trainloader.dataset)


def test(model, testloader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, _ in testloader:
            x = x.to(device)
            x_hat = model.decoder(model.encoder(x))
            loss = ((x - x_hat) ** 2).sum() + model.encoder.kl
            val_loss += loss.item() / x.shape[0]
    return val_loss / len(testloader.dataset)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args.log_dir = os.path.join(args.log_dir, timestamp)
    writer = SummaryWriter(args.log_dir)

    input_transform = transforms.Compose([
        transforms.Resize((80, 160)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset = CarlaDataset(root=args.data_dir, is_depth=args.is_depth, transform=input_transform)

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.seed(args.random_seed)
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=4)

    model = VariationalAutoencoder(
        in_channels=1 if args.is_depth else 3,
        latent_dims=args.latent_space
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, optimizer, device)
        writer.add_scalar("Training Loss/epoch", train_loss, epoch + 1)
        val_loss = test(model, val_loader, device)
        writer.add_scalar("Validation Loss/epoch", val_loss, epoch + 1)
        print(f'Epoch {epoch + 1}/{args.num_epochs} \t train loss {train_loss:.3f} \t val loss {val_loss:.3f}')

    torch.save(model.state_dict(), os.path.join(args.log_dir, f'ae_{dataset.mode}.pth'))
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Variational Autoencoder on Carla Dataset")
    parser.add_argument("--data_dir", type=str, default="data_160x80", help="Path to the dataset directory")
    parser.add_argument("--is_depth", action="store_true", help="Use depth images instead of semseg images")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--latent_space", type=int, default=50, help="Dimensionality of the latent space")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--log_dir", type=str, default="results/autoencoder-semseg", help="Directory to save training logs and models")

    args = parser.parse_args()
    main(args)
