import numpy as np
import os

def generate_dummy_files(output_base_dir="dummy_dataset",
                         num_file_pairs=2,
                         samples_per_file=10,
                         input_shape=(5, 1000, 70),  # C, T, G
                         target_shape=(70, 70),   # H, W
                         data_type=np.float32):
    """
    Generates dummy .npy data files and a metadata list file.

    Args:
        output_base_dir (str): Base directory to save the dummy data and metadata.
        num_file_pairs (int): Number of (seismic_data, velocity_model) .npy file pairs to create.
        samples_per_file (int): Number of samples (S) within each .npy file.
        input_shape (tuple): Shape of a single input sample (channels, time_steps, geophones).
        target_shape (tuple): Shape of a single target sample (height, width).
        data_type (np.dtype): NumPy data type for the array elements.
    """

    # Create directories
    data_dir = os.path.join(output_base_dir, "data")
    model_dir = os.path.join(output_base_dir, "model")
    metadata_dir = os.path.join(output_base_dir, "file_lists") # For metadata .txt files

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    print(f"Generating dummy data in: {os.path.abspath(output_base_dir)}")

    metadata_entries = []

    for i in range(num_file_pairs):
        # --- Generate Input Data (Seismic) ---
        # Shape: (S, C, T, G)
        dummy_input_data = np.random.rand(samples_per_file, *input_shape).astype(data_type)
        # Scale to a common seismic data range e.g., -1 to 1 (optional, random is often fine for schema tests)
        dummy_input_data = (dummy_input_data * 2) - 1 
        input_filename = f"dummy_data_{i}.npy"
        input_filepath = os.path.join(data_dir, input_filename)
        np.save(input_filepath, dummy_input_data)
        print(f"Saved: {input_filepath} with shape {dummy_input_data.shape}")

        # --- Generate Target Data (Velocity Model) ---
        # Shape: (S, H, W)
        dummy_target_data = np.random.rand(samples_per_file, *target_shape).astype(data_type)
        # Scale to a plausible velocity range (e.g., 1500 to 4500 m/s)
        # This matches the baseline model's output scaling: output * 1500 + 3000
        dummy_target_data = dummy_target_data * 3000 + 1500 
        target_filename = f"dummy_model_{i}.npy"
        target_filepath = os.path.join(model_dir, target_filename)
        np.save(target_filepath, dummy_target_data)
        print(f"Saved: {target_filepath} with shape {dummy_target_data.shape}")

        # Store relative paths for the metadata file, making it portable
        # The data loaders in OpenFWI often take paths relative to where the script/config is,
        # or expect full paths. For this dummy setup, let's use paths relative to the metadata file's dir parent.
        # So if metadata is in dummy_dataset/file_lists/dummy_train.txt
        # and data is in dummy_dataset/data/dummy_data_0.npy
        # the path could be ../data/dummy_data_0.npy
        # However, the tutorial uses absolute-like paths for Kaggle.
        # For simplicity and clarity with local dummy data, let's store paths relative to output_base_dir.
        # The data loading script will need to prepend output_base_dir or an absolute path.
        # OR, for max compatibility with tutorial scripts that might expect full paths in the .txt:
        abs_target_filepath = os.path.abspath(target_filepath)
        abs_input_filepath = os.path.abspath(input_filepath)
        metadata_entries.append(f"{abs_target_filepath} {abs_input_filepath}")


    # --- Create Metadata File (e.g., for training) ---
    # This file will list pairs of (model_path, data_path)
    metadata_filename = "dummy_train.txt"
    metadata_filepath = os.path.join(metadata_dir, metadata_filename)
    with open(metadata_filepath, "w") as f:
        for entry in metadata_entries:
            f.write(entry + "\n")
    print(f"Saved metadata list: {metadata_filepath}")

    print("\nGeneration complete.")
    print(f"To use this data, your data loader should read '{metadata_filepath}'")
    print("and interpret the paths accordingly (they are currently absolute).")
    print("You might need to adjust your data loading scripts if they expect paths relative to a different root.")

if __name__ == "__main__":
    generate_dummy_files(
        output_base_dir="Yale_UNC-CH_Geophysical_Waveform_Inversion/dummy_local_dataset", # Save inside project
        num_file_pairs=2,       # Create 2 pairs of (data, model) files
        samples_per_file=5,     # Each .npy file will contain 5 samples
        data_type=np.float32    # Use float32 for dummy data
    )
    print("\nTo run this script, navigate to the root of your 'MyKaggleProject' workspace")
    print("and execute: python Yale_UNC-CH_Geophysical_Waveform_Inversion/utils/generate_dummy_data.py") 