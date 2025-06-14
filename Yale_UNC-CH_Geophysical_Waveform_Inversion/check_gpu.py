import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import sys

# Since this script is in the project root, we need to add 'src' to the path
# to allow for relative imports of our modules (model, dataset).
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from model import UNet, ForwardModel
from dataset import SeismicDataset


def check_pytorch_gpu():
    """
    Checks and prints the status of PyTorch's GPU availability.
    """
    print("--- PyTorch GPU Check ---")
    print(f"PyTorch Version: {torch.__version__}")
    is_available = torch.cuda.is_available()
    print(f"CUDA Available: {is_available}")
    
    if is_available:
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("PyTorch cannot find a CUDA-enabled GPU. Training will run on CPU.")
    print("-------------------------\n")
    return is_available

def train_with_cycle_consistency():
    """
    Main training function, adapted to run from the root directory.
    """
    # =====================================================================
    # 1. Hyperparameters & Setup
    # =====================================================================
    data_directory = 'data/train_samples/'
    num_samples_to_test = 32
    learning_rate = 1e-4
    num_epochs = 5
    batch_size = 4
    lambda_cycle = 0.5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    # =====================================================================
    # 2. Data Loading
    # =====================================================================
    print("--- Loading data ---")
    full_dataset = SeismicDataset(data_dir=data_directory)
    print(f"Found {len(full_dataset)} total samples in the dataset.")
    indices = np.arange(num_samples_to_test)
    test_subset = Subset(full_dataset, indices)
    print(f"Using a subset of {len(test_subset)} samples for this test run.")
    train_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=True)
    
    # =====================================================================
    # 3. Model, Loss, and Optimizer Initialization
    # =====================================================================
    print("--- Initializing models, loss functions, and optimizer ---")
    inversion_model = UNet(in_channels=5, out_channels=1).to(device)
    forward_model = ForwardModel(in_channels=1, out_channels=5).to(device)
    primary_loss_fn = nn.L1Loss()
    cycle_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(
        list(inversion_model.parameters()) + list(forward_model.parameters()),
        lr=learning_rate
    )

    # =====================================================================
    # 4. The Training Loop
    # =====================================================================
    print("--- Starting training test ---")
    inversion_model.train()
    forward_model.train()

    for epoch in range(num_epochs):
        for i, (real_seismic, real_map) in enumerate(train_loader):
            real_seismic = real_seismic.to(device)
            real_map = real_map.to(device)
            predicted_map = inversion_model(real_seismic)
            reconstructed_seismic = forward_model(predicted_map)
            loss_primary = primary_loss_fn(predicted_map, real_map)
            loss_cycle = cycle_loss_fn(reconstructed_seismic, real_seismic)
            total_loss = loss_primary + lambda_cycle * loss_cycle
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Total Loss: {total_loss.item():.4f} | "
              f"Primary MAE Loss: {loss_primary.item():.4f} | "
              f"Cycle MSE Loss: {loss_cycle.item():.4f}")

    print("--- Training test finished successfully! ---")


if __name__ == '__main__':
    # First, check for GPU
    gpu_ok = check_pytorch_gpu()
    
    # Then, proceed to run the training test
    if gpu_ok:
        print("\nProceeding to training test...")
        train_with_cycle_consistency()
    else:
        print("\nSkipping training test due to no GPU being available.") 