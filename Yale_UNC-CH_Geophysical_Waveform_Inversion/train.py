import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import logging

# It's good practice to import from your own project modules.
# We assume you run this script from the root of the 'Yale_UNC-CH_Geophysical_Waveform_Inversion' directory.
from src.model import UNet, UNetV2, UNetWithAttention, TransformerGenerator, ForwardModel
from src.dataset import SeismicDataset

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(args):
    """
    Main training function.
    """
    # --- Device Configuration ---
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logging.info(f"Using device: {device}")

    # --- Data Loading ---
    # AZURE-SPECIFIC HOOK: Here you might download your dataset from Azure Blob Storage or ADLS
    # to the path specified in --data_dir if it doesn't exist locally.
    logging.info("Loading data...")
    try:
        full_dataset = SeismicDataset(data_dir=args.data_dir)
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Error initializing dataset: {e}")
        logging.error("Please ensure the --data_dir argument points to a directory with 'data' and 'model' subfolders.")
        return

    # --- Train/Validation Split ---
    n_total = len(full_dataset)
    n_val = int(n_total * args.val_percent)
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val],
                                              generator=torch.Generator().manual_seed(42))

    # Apply max sample limits if specified
    if args.max_train_samples > 0:
        train_indices = list(range(min(args.max_train_samples, len(train_dataset))))
        train_dataset = Subset(train_dataset, train_indices)
    if args.max_val_samples > 0:
        val_indices = list(range(min(args.max_val_samples, len(val_dataset))))
        val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    logging.info(f"Data loaded: {n_train} training samples, {n_val} validation samples.")

    # --- Model, Optimizer, Loss ---
    model_dict = {
        'UNet': UNet,
        'UNetV2': UNetV2,
        'UNetWithAttention': UNetWithAttention,
        'TransformerGenerator': TransformerGenerator
    }
    if args.model_name not in model_dict:
        raise ValueError(f"Model '{args.model_name}' not recognized. Choose from: {list(model_dict.keys())}")
    model = model_dict[args.model_name](in_channels=5, out_channels=1).to(device)
    logging.info(f"Model {args.model_name} created. Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # The competition metric is Mean Absolute Error (MAE), which corresponds to L1Loss in PyTorch.
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Optional: Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True)

    # Optional cycle-consistency
    use_cycle = args.cycle_consistency
    lambda_cycle = args.lambda_cycle
    if use_cycle:
        forward_model = ForwardModel(in_channels=1, out_channels=5).to(device)
        cycle_criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(list(model.parameters()) + list(forward_model.parameters()), lr=args.lr)
        logging.info(f"Cycle-consistency enabled (lambda_cycle={lambda_cycle})")

    # --- Training Loop ---
    best_val_loss = float('inf')
    # AZURE-SPECIFIC HOOK: You can use azureml.core.Run.get_context() to log metrics.
    # run = Run.get_context()

    for epoch in range(args.epochs):
        model.train()
        if use_cycle:
            forward_model.train()
        epoch_loss = 0
        epoch_cycle_loss = 0
        
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss = loss
                if use_cycle:
                    reconstructed_inputs = forward_model(outputs)
                    cycle_loss = cycle_criterion(reconstructed_inputs, inputs)
                    total_loss = loss + lambda_cycle * cycle_loss
                    epoch_cycle_loss += cycle_loss.item()
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                pbar.update(inputs.size(0))
                if use_cycle:
                    pbar.set_postfix(**{'loss (batch)': loss.item(), 'cycle (batch)': epoch_cycle_loss})
                else:
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
        
        avg_train_loss = epoch_loss / len(train_loader)
        avg_cycle_loss = epoch_cycle_loss / len(train_loader) if use_cycle else None
        logging.info(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")
        if use_cycle:
            logging.info(f"Epoch {epoch + 1} - Average Cycle Loss: {avg_cycle_loss:.4f}")

        # --- Validation Loop ---
        val_loss = evaluate(model, val_loader, device, criterion)
        logging.info(f'Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}')
        
        # Update learning rate
        scheduler.step(val_loss)

        # AZURE-SPECIFIC HOOK: Log metrics to your Azure run
        # run.log('train_loss', avg_train_loss)
        # run.log('val_loss', val_loss)

        # --- Save Best Model ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.save_dir, exist_ok=True)
            model_filename = f"{args.model_name}_bs{args.batch_size}_lr{args.lr}_e{args.epochs}"
            if use_cycle:
                model_filename += f"_cycle{lambda_cycle}"
            model_filename += ".pth"
            model_path = os.path.join(args.save_dir, model_filename)
            torch.save(model.state_dict(), model_path)
            logging.info(f"New best model saved to {model_path} (val_loss: {best_val_loss:.4f})")

    logging.info("Training finished.")
    # AZURE-SPECIFIC HOOK: The saved model in args.save_dir will automatically be uploaded
    # to your Azure ML run's outputs if you configure the job correctly.

def evaluate(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train a model for seismic inversion.")
    parser.add_argument('--model_name', type=str, default='UNet', help='Model architecture: UNet, UNetV2, UNetWithAttention, TransformerGenerator')
    parser.add_argument('--epochs', type=int, default=25, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Input batch size for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the training data.')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints.')
    parser.add_argument('--val_percent', type=float, default=0.15, help='Percentage of data to use for validation.')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading.')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--cycle_consistency', action='store_true', default=False, help='Enable cycle-consistency loss (physics loss).')
    parser.add_argument('--lambda_cycle', type=float, default=0.5, help='Weight for cycle-consistency loss.')
    parser.add_argument('--max_train_samples', type=int, default=0, help='Max number of training samples (0 for all)')
    parser.add_argument('--max_val_samples', type=int, default=0, help='Max number of validation samples (0 for all)')
    
    args = parser.parse_args()
    train_model(args)

if __name__ == '__main__':
    main() 