import os
import torch
from torch.utils.data import Dataset
import numpy as np
import glob

class SeismicDataset(Dataset):
    """
    Custom PyTorch Dataset for loading the seismic and velocity map data.
    This dataset handles .npy files that contain a batch of samples.
    It maps a global index to a specific file and a specific slice within that file.
    """
    def __init__(self, data_dir, samples_per_file=500, transform=None):
        """
        Args:
            data_dir (string): Directory containing the training data subfolders.
            samples_per_file (int): The number of samples (S) in each .npy file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples_per_file = samples_per_file
        self.file_pairs = []

        print(f"DEBUG: Starting recursive search in {data_dir}")
        for root, _, files in os.walk(data_dir):
            print(f"DEBUG: Checking directory: {root}")
            print(f"DEBUG: Files: {files}")
            seismic_files = sorted([f for f in files if f.startswith('seis') and f.endswith('.npy')])
            for seismic_file in seismic_files:
                seismic_path = os.path.join(root, seismic_file)
                velocity_filename = seismic_file.replace('seis', 'vel', 1)
                velocity_path = os.path.join(root, velocity_filename)
                print(f"DEBUG: Trying pair: {seismic_path}, {velocity_path} (exists: {os.path.exists(velocity_path)})")
                if os.path.exists(velocity_path):
                    self.file_pairs.append((seismic_path, velocity_path))
        print(f"DEBUG: Total pairs found: {len(self.file_pairs)}")
        if self.file_pairs:
            print('DEBUG: First few pairs:', self.file_pairs[:3])
        
        if not self.file_pairs:
            raise FileNotFoundError(f"No matching 'seis' and 'vel' file pairs found in {data_dir}")

        self.total_samples = len(self.file_pairs) * self.samples_per_file

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Determine which file and which sample in that file to load
        file_index = idx // self.samples_per_file
        sample_index_in_file = idx % self.samples_per_file

        seismic_path, velocity_path = self.file_pairs[file_index]
        
        # Load the batch from the .npy file and select the specific sample
        seismic_data = np.load(seismic_path)[sample_index_in_file]
        velocity_map = np.load(velocity_path)[sample_index_in_file]
        
        sample = {'seismic': seismic_data, 'velocity': velocity_map}

        if self.transform:
            sample = self.transform(sample)
            
        # Convert numpy arrays to torch tensors
        sample['seismic'] = torch.from_numpy(sample['seismic']).float()
        sample['velocity'] = torch.from_numpy(sample['velocity']).float()

        return sample['seismic'], sample['velocity']


# This main block is for testing the dataset implementation
if __name__ == '__main__':
    print("--- Testing SeismicDataset ---")
    # This path is relative to the project root (MyKaggleProject)
    test_data_dir = './Yale_UNC-CH_Geophysical_Waveform_Inversion/data/train_samples/'
    SAMPLES_PER_FILE = 500 # Based on the last run's output
    
    try:
        dataset = SeismicDataset(data_dir=test_data_dir, samples_per_file=SAMPLES_PER_FILE)
        print(f"Successfully found {len(dataset.file_pairs)} file pairs.")
        print(f"Total samples available: {len(dataset)}")
        
        # Get a sample
        print("Fetching a sample from the dataset...")
        seismic_sample, velocity_sample = dataset[0] # Get the first sample
        
        # Check shapes and types
        print(f"Sample seismic shape: {seismic_sample.shape}, type: {seismic_sample.dtype}")
        print(f"Sample velocity shape: {velocity_sample.shape}, type: {velocity_sample.dtype}")
        
        # The shape should now be for a single sample, not a batch
        assert seismic_sample.shape == (5, 1000, 70)
        assert velocity_sample.shape == (1, 70, 70)
        assert seismic_sample.dtype == torch.float32
        
        # Fetch another sample to test indexing
        print("Fetching another sample (index 501)...")
        seismic_sample_2, velocity_sample_2 = dataset[501]
        print(f"Sample 501 seismic shape: {seismic_sample_2.shape}")
        
        assert seismic_sample_2.shape == (5, 1000, 70)

        print("\n--- SeismicDataset test passed! ---")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the data directory exists and is populated.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")