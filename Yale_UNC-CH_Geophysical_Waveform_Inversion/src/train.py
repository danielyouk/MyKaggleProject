import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import sys
import os
import argparse
from datetime import datetime

# Add the 'src' directory to the Python path
# This allows us to run the script from the project root (MyKaggleProject)
# and still import modules from the 'src' directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import model
from dataset import SeismicDataset

# Add import for InversionNet
sys.path.append(os.path.join(os.path.dirname(__file__), '../OpenFWI'))
from src.network import InversionNet

# #################################################################################
# This script contains two training pipelines:
# 1. train_cycle_consistency: A generator model with a forward model for physics loss.
# 2. train_gan: A Pix2Pix-style Generative Adversarial Network.
# #################################################################################

def train_cycle_consistency(model_name, experiment_name):
    """
    Main training function with experiment tracking and checkpointing,
    using the cycle-consistency (forward model) method.
    """
    # =====================================================================
    # 1. Hyperparameters & Setup
    # =====================================================================
    # --- Paths and Data Config ---
    data_directory = 'data/train_samples/'
    samples_per_file = 500
    num_samples_to_test = 32
    
    # --- Training Config ---
    learning_rate = 1e-4
    num_epochs = 15
    batch_size = 4
    lambda_cycle = 0.5
    
    # --- Experiment Tracking Setup ---
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    # If no experiment name is provided, use the model name.
    if experiment_name is None:
        experiment_name = model_name
    run_name = f"{experiment_name}_{timestamp}"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_dir = os.path.abspath(os.path.join(base_dir, '..', 'experiments', run_name))
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    logs_dir = os.path.join(experiment_dir, 'logs')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=logs_dir)
    print(f"--- Starting Experiment: {run_name} ---")
    print(f"Checkpoints will be saved in: {checkpoints_dir}")
    print(f"TensorBoard logs will be saved in: {logs_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    # =====================================================================
    # 2. Data Loading
    # =====================================================================
    print("--- Loading data ---")
    full_dataset = SeismicDataset(
        data_dir=data_directory,
        samples_per_file=samples_per_file
    )
    print(f"Found {len(full_dataset.file_pairs)} data files, with {len(full_dataset)} total samples.")
    print(f"Found {len(full_dataset.file_pairs)} seis/vel pairs.")
    if full_dataset.file_pairs:
        print('First few pairs:', full_dataset.file_pairs[:3])
    
    indices = np.arange(num_samples_to_test)
    test_subset = Subset(full_dataset, indices)
    print(f"Using a subset of {len(test_subset)} samples for this test run.")
    
    train_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=True)
    
    # =====================================================================
    # 3. Model, Loss, and Optimizer Initialization
    # =====================================================================
    print(f"--- Initializing model: {model_name} ---")
    
    # Dynamically get the model class from the 'model' module
    if model_name == "InversionNet":
        from src.network import InversionNet
        model_class = InversionNet
    else:
        model_class = getattr(model, model_name)
    
    inversion_model = model_class(in_channels=5, out_channels=1).to(device)
    forward_model = model.ForwardModel(in_channels=1, out_channels=5).to(device)
    primary_loss_fn = nn.L1Loss() # MAE Loss (Competition Metric)
    cycle_loss_fn = nn.MSELoss()
    optimizer = optim.Adam(
        list(inversion_model.parameters()) + list(forward_model.parameters()),
        lr=learning_rate
    )

    # =====================================================================
    # 4. The Training Loop
    # =====================================================================
    print("--- Starting training ---")
    best_primary_loss = float('inf')
    inversion_model.train()
    forward_model.train()

    for epoch in range(num_epochs):
        epoch_total_loss = 0.0
        epoch_primary_loss = 0.0
        epoch_cycle_loss = 0.0
        
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
            
            epoch_total_loss += total_loss.item()
            epoch_primary_loss += loss_primary.item()
            epoch_cycle_loss += loss_cycle.item()
        
        avg_total_loss = epoch_total_loss / len(train_loader)
        avg_primary_loss = epoch_primary_loss / len(train_loader)
        avg_cycle_loss = epoch_cycle_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Avg Total Loss: {avg_total_loss:.4f} | Competition MAE: {avg_primary_loss:.4f}")

        # --- Logging to TensorBoard ---
        writer.add_scalar('Loss/Total_Avg', avg_total_loss, epoch)
        writer.add_scalar('Loss/Primary_MAE_Avg', avg_primary_loss, epoch)
        writer.add_scalar('Loss/Cycle_MSE_Avg', avg_cycle_loss, epoch)
        if device.type == 'cuda':
            gpu_mem_gb = torch.cuda.memory_allocated(0) / (1024**3)
            writer.add_scalar('GPU/Memory_Allocated_GB', gpu_mem_gb, epoch)

        # --- Model Checkpointing (based on Primary MAE Loss) ---
        if avg_primary_loss < best_primary_loss:
            best_primary_loss = avg_primary_loss
            checkpoint_filename = f'{experiment_name}_bs{batch_size}_lr{learning_rate:.6f}_e{num_epochs}_cycle{lambda_cycle:.2f}.pth'
            checkpoint_path = os.path.join(checkpoints_dir, checkpoint_filename)
            torch.save(inversion_model.state_dict(), checkpoint_path)
            print(f"  -> New best model saved with Competition MAE: {best_primary_loss:.4f} at {checkpoint_path}")

    print("--- Training finished successfully! ---")
    writer.close()
    print(f"\nTo view logs, run the following command in your terminal:\n  tensorboard --logdir experiments/{run_name}/logs")

    # After training loop
    for fname in os.listdir(checkpoints_dir):
        if fname != checkpoint_filename:
            os.remove(os.path.join(checkpoints_dir, fname))


def train_gan(generator_model_name, experiment_name):
    """
    Trains a Generative Adversarial Network (Pix2Pix style).
    The Discriminator's job is to tell real maps from fake ones.
    """
    # =====================================================================
    # 1. Hyperparameters & Setup
    # =====================================================================
    data_directory = 'data/train_samples/'
    samples_per_file = 500
    num_samples_to_test = 32
    
    learning_rate = 2e-4
    num_epochs = 50 # GANs often need more epochs to stabilize
    batch_size = 1   # Pix2Pix GANs often perform best with a batch size of 1
    lambda_l1 = 100  # Weight for the L1 reconstruction loss (primary goal)
    
    # --- Experiment Tracking Setup ---
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    if experiment_name is None:
        experiment_name = f"{generator_model_name}_GAN"
    run_name = f"{experiment_name}_{timestamp}"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_dir = os.path.abspath(os.path.join(base_dir, '..', 'experiments', run_name))
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    logs_dir = os.path.join(experiment_dir, 'logs')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=logs_dir)
    print(f"--- Starting GAN Experiment: {run_name} ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    # =====================================================================
    # 2. Data Loading
    # =====================================================================
    print("--- Loading data ---")
    full_dataset = SeismicDataset(data_dir=data_directory, samples_per_file=samples_per_file)
    indices = np.arange(num_samples_to_test)
    test_subset = Subset(full_dataset, indices)
    train_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=True)
    print(f"Using a subset of {len(test_subset)} samples for this test run.")

    # =====================================================================
    # 3. Model, Loss, and Optimizer Initialization
    # =====================================================================
    print(f"--- Initializing GAN models ---")
    try:
        gen_model_class = getattr(model, generator_model_name)
        generator = gen_model_class(in_channels=5, out_channels=1).to(device)
    except AttributeError:
        raise ValueError(f"Generator model '{generator_model_name}' not found in 'src/model.py'.")
    
    discriminator = model.Discriminator(in_channels=1).to(device)
    
    # Loss Functions
    adversarial_loss_fn = nn.MSELoss()
    l1_loss_fn = nn.L1Loss()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # =====================================================================
    # 4. The GAN Training Loop
    # =====================================================================
    print("--- Starting GAN training ---")
    best_l1_loss = float('inf')

    for epoch in range(num_epochs):
        for i, (real_seismic, real_map) in enumerate(train_loader):
            real_seismic = real_seismic.to(device)
            real_map = real_map.to(device)

            # Adversarial ground truths (for a 4x4 PatchGAN output)
            patch_size = (1, 1, 4, 4)
            valid = torch.ones(real_map.size(0), *patch_size[1:], device=device)
            fake = torch.zeros(real_map.size(0), *patch_size[1:], device=device)

            # --- Train Generator ---
            optimizer_G.zero_grad()
            fake_map = generator(real_seismic)
            loss_G_adv = adversarial_loss_fn(discriminator(fake_map), valid)
            loss_G_l1 = l1_loss_fn(fake_map, real_map)
            loss_G = loss_G_adv + lambda_l1 * loss_G_l1
            loss_G.backward()
            optimizer_G.step()

            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            loss_D_real = adversarial_loss_fn(discriminator(real_map), valid)
            loss_D_fake = adversarial_loss_fn(discriminator(fake_map.detach()), fake)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

        # --- Logging and Checkpointing ---
        print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f} | G MAE (L1): {loss_G_l1.item():.4f}")
        
        writer.add_scalar('GAN/Loss_D', loss_D.item(), epoch)
        writer.add_scalar('GAN/Loss_G', loss_G.item(), epoch)
        writer.add_scalar('GAN/Loss_G_L1', loss_G_l1.item(), epoch)

        if loss_G_l1.item() < best_l1_loss:
            best_l1_loss = loss_G_l1.item()
            metric_name = 'L1'
            checkpoint_filename = f'{experiment_name}_bs{batch_size}_lr{learning_rate:.6f}_e{num_epochs}_cycle{lambda_l1:.2f}_{metric_name}{best_l1_loss:.2f}.pth'
            checkpoint_path = os.path.join(checkpoints_dir, checkpoint_filename)
            torch.save(generator.state_dict(), checkpoint_path)
            print(f"  -> New best generator saved with {metric_name} loss: {best_l1_loss:.4f} at {checkpoint_path}")

    print("--- GAN Training finished successfully! ---")
    writer.close()
    print(f"\nTo view logs, run the following command in your terminal:\n  tensorboard --logdir {logs_dir}")

    # After training loop
    for fname in os.listdir(checkpoints_dir):
        if fname != checkpoint_filename:
            os.remove(os.path.join(checkpoints_dir, fname))


def train_fine_tune_base_model(experiment_name, base_model_path, num_epochs=15, learning_rate=1e-5):
    """
    Fine-tune the base model (fvb_l1.pth) using the cycle-consistency approach.
    This function assumes the base model is a UNet architecture.
    """
    # =====================================================================
    # 1. Hyperparameters & Setup
    # =====================================================================
    # --- Paths and Data Config ---
    data_directory = 'data/train_samples/'
    samples_per_file = 500
    num_samples_to_test = 32
    
    # --- Training Config ---
    batch_size = 4
    lambda_cycle = 0.5
    
    # --- Experiment Tracking Setup ---
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    if experiment_name is None:
        experiment_name = "fine_tuned_base"
    run_name = f"{experiment_name}_{timestamp}"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_dir = os.path.abspath(os.path.join(base_dir, '..', 'experiments', run_name))
    logs_dir = os.path.join(experiment_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    # Central checkpoints directory
    central_checkpoints_dir = os.path.abspath(os.path.join(base_dir, '..', 'experiments', 'checkpoints'))
    os.makedirs(central_checkpoints_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir=logs_dir)
    print(f"--- Starting Fine-tuning Experiment: {run_name} ---")
    print(f"TensorBoard logs will be saved in: {logs_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using device: {device} ---")

    # =====================================================================
    # 2. Data Loading
    # =====================================================================
    print("--- Loading data ---")
    full_dataset = SeismicDataset(
        data_dir=data_directory,
        samples_per_file=samples_per_file
    )
    print(f"Found {len(full_dataset.file_pairs)} data files, with {len(full_dataset)} total samples.")
    print(f"Found {len(full_dataset.file_pairs)} seis/vel pairs.")
    if full_dataset.file_pairs:
        print('First few pairs:', full_dataset.file_pairs[:3])
    
    indices = np.arange(num_samples_to_test)
    test_subset = Subset(full_dataset, indices)
    print(f"Using a subset of {len(test_subset)} samples for this test run.")
    
    train_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=True)
    
    # =====================================================================
    # 3. Model, Loss, and Optimizer Initialization
    # =====================================================================
    print("--- Loading base model and initializing for fine-tuning ---")
    
    # Initialize model and load base weights
    inversion_model = InversionNet().to(device)
    try:
        checkpoint = torch.load(base_model_path)
        inversion_model.load_state_dict(checkpoint['model'])
        print(f"Successfully loaded base model weights from {base_model_path}")
    except Exception as e:
        print(f"Error loading base model: {e}")
        return
    
    forward_model = model.ForwardModel(in_channels=1, out_channels=5).to(device)
    primary_loss_fn = nn.L1Loss() # MAE Loss (Competition Metric)
    cycle_loss_fn = nn.MSELoss()
    
    # Use a smaller learning rate for fine-tuning
    optimizer = optim.Adam(
        list(inversion_model.parameters()) + list(forward_model.parameters()),
        lr=learning_rate
    )

    # =====================================================================
    # 4. The Training Loop
    # =====================================================================
    print("--- Starting fine-tuning ---")
    best_primary_loss = float('inf')
    inversion_model.train()
    forward_model.train()

    for epoch in range(num_epochs):
        epoch_total_loss = 0.0
        epoch_primary_loss = 0.0
        epoch_cycle_loss = 0.0
        
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
            
            epoch_total_loss += total_loss.item()
            epoch_primary_loss += loss_primary.item()
            epoch_cycle_loss += loss_cycle.item()
        
        avg_total_loss = epoch_total_loss / len(train_loader)
        avg_primary_loss = epoch_primary_loss / len(train_loader)
        avg_cycle_loss = epoch_cycle_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Avg Total Loss: {avg_total_loss:.4f} | Competition MAE: {avg_primary_loss:.4f}")

        # --- Logging to TensorBoard ---
        writer.add_scalar('Loss/Total_Avg', avg_total_loss, epoch)
        writer.add_scalar('Loss/Primary_MAE_Avg', avg_primary_loss, epoch)
        writer.add_scalar('Loss/Cycle_MSE_Avg', avg_cycle_loss, epoch)
        if device.type == 'cuda':
            gpu_mem_gb = torch.cuda.memory_allocated(0) / (1024**3)
            writer.add_scalar('GPU/Memory_Allocated_GB', gpu_mem_gb, epoch)

        # --- Model Checkpointing (based on Primary MAE Loss) ---
        if avg_primary_loss < best_primary_loss:
            best_primary_loss = avg_primary_loss
            checkpoint_filename = f'{experiment_name}_bs{batch_size}_lr{learning_rate:.6f}_e{num_epochs}_cycle{lambda_cycle:.2f}.pth'
            checkpoint_path = os.path.join(central_checkpoints_dir, checkpoint_filename)
            torch.save(inversion_model.state_dict(), checkpoint_path)
            print(f"  -> New best model saved with Competition MAE: {best_primary_loss:.4f} at {checkpoint_path}")

    print("--- Fine-tuning finished successfully! ---")
    writer.close()
    print(f"\nTo view logs, run the following command in your terminal:\n  tensorboard --logdir experiments/{run_name}/logs")

    # After training loop
    for fname in os.listdir(central_checkpoints_dir):
        if not fname.startswith(f'{experiment_name}_bs{batch_size}_lr{learning_rate:.6f}_e{num_epochs}_cycle{lambda_cycle:.2f}'):
            os.remove(os.path.join(central_checkpoints_dir, fname))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train FWI models')
    parser.add_argument('--mode', type=str, required=True, choices=['cycle', 'gan', 'fine_tune'],
                      help='Training mode: cycle (cycle-consistency), gan (GAN), or fine_tune (fine-tune base model)')
    parser.add_argument('--model_name', type=str, required=False,
                      help='Model name for cycle or GAN training')
    parser.add_argument('--experiment_name', type=str, required=False,
                      help='Name for the experiment')
    parser.add_argument('--base_model_path', type=str, required=False,
                      help='Path to base model for fine-tuning')
    parser.add_argument('--num_epochs', type=int, default=15,
                      help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate for training')
    
    args = parser.parse_args()
    
    if args.mode == 'cycle':
        if not args.model_name:
            raise ValueError("Model name is required for cycle training mode")
        train_cycle_consistency(args.model_name, args.experiment_name)
    elif args.mode == 'gan':
        if not args.model_name:
            raise ValueError("Model name is required for GAN training mode")
        train_gan(args.model_name, args.experiment_name)
    elif args.mode == 'fine_tune':
        if not args.base_model_path:
            raise ValueError("Base model path is required for fine-tuning mode")
        train_fine_tune_base_model(
            args.experiment_name,
            args.base_model_path,
            args.num_epochs,
            args.learning_rate
        ) 