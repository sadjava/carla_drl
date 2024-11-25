import torch
import time
from typing import Tuple

from carla_drl.semantic_segmentation.unet import UNet
from carla_drl.autoencoder.model import VariationalAutoencoder
from carla_drl.lane_following.model import ActorCritic

# Function to measure inference time
def measure_inference_time(model, input_tensor, device, num_runs=5):
    model.to(device)
    model.eval()
    times = []
    
    # Warm-up runs
    for _ in range(2):
        with torch.no_grad():
            if isinstance(input_tensor, Tuple):
                new_input_tensor = (torch.randn_like(input_tensor[0]), torch.randn_like(input_tensor[1]), 
                                   torch.randn_like(input_tensor[2]))
                model((new_input_tensor[0].to(device), new_input_tensor[1].to(device), new_input_tensor[2].to(device)))
            else:
                new_input_tensor = torch.randn_like(input_tensor)
                model(new_input_tensor.to(device))
    
    # Actual measurement runs
    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            if isinstance(input_tensor, Tuple):
                new_input_tensor = (torch.randn_like(input_tensor[0]), torch.randn_like(input_tensor[1]), 
                                   torch.randn_like(input_tensor[2]))
                model((new_input_tensor[0].to(device), new_input_tensor[1].to(device), new_input_tensor[2].to(device)))
            else:
                new_input_tensor = torch.randn_like(input_tensor)
                model(new_input_tensor.to(device))
        end_time = time.time()
        times.append(end_time - start_time)
    
    return sum(times) / len(times)  # Return average time

# Initialize models
semantic_model = UNet(num_classes=29)
vae_model = VariationalAutoencoder(latent_dims=95)
ac_model = ActorCritic(obs_dim=100, action_dim=2, action_std_init=0.2)

# Create random input tensors
semantic_input = torch.randn(1, 3, 80, 160)  # For UNet
vae_input = torch.randn(1, 3, 80, 160)     # For VAE
ac_input = torch.randn(1, 100) # For AC


# Measure sizes
print(f"Semantic Segmentation Model Size: {sum(p.numel() for p in semantic_model.parameters())} parameters")
print(f"Autoencoder Model Size: {sum(p.numel() for p in vae_model.parameters())} parameters")
print(f"Lane Following Model Size: {sum(p.numel() for p in ac_model.parameters())} parameters")

# Measure inference time on CPU
cpu_time_semantic = measure_inference_time(semantic_model, semantic_input, 'cpu')
cpu_time_vae = measure_inference_time(vae_model, vae_input, 'cpu')
cpu_time_lane = measure_inference_time(ac_model, ac_input, 'cpu')

print(f"Average CPU Inference Time - Semantic Segmentation: {cpu_time_semantic:.6f} seconds")
print(f"Average CPU Inference Time - Autoencoder: {cpu_time_vae:.6f} seconds")
print(f"Average CPU Inference Time - Lane Following: {cpu_time_lane:.6f} seconds")
print(f"Average CPU Inference FPS - {1 / (cpu_time_semantic + cpu_time_vae + cpu_time_lane)}")

# Measure inference time on CUDA if available
if torch.cuda.is_available():
    cuda_time_semantic = measure_inference_time(semantic_model, semantic_input, 'cuda')
    cuda_time_vae = measure_inference_time(vae_model, vae_input, 'cuda')
    cuda_time_lane = measure_inference_time(ac_model, ac_input, 'cuda')

    print(f"Average CUDA Inference Time - Semantic Segmentation: {cuda_time_semantic:.6f} seconds")
    print(f"Average CUDA Inference Time - Autoencoder: {cuda_time_vae:.6f} seconds")
    print(f"Average CUDA Inference Time - Lane Following: {cuda_time_lane:.6f} seconds")
    print(f"Average CUDA Inference FPS - {1 / (cuda_time_semantic + cuda_time_vae + cuda_time_lane)}")
