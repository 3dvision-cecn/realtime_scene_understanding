import os
import shutil
from pathlib import Path

def split_and_rename_data(parent_dir: str, train_dir: str, test_dir: str, train_split_ratio: float = 0.8):
    """
    Splits .h5ad files from subdirectories into training and testing sets,
    renames them, and copies them to new destination folders.

    Args:
        parent_dir (str): The path to the main directory containing subfolders (e.g., '01', '02').
        train_dir (str): The path to the destination directory for training files.
        test_dir (str): The path to the destination directory for testing files.
        train_split_ratio (float): The proportion of files to be included in the training set.
                                   Defaults to 0.8 (80%).
    """
    # --- 1. Setup ---
    # Convert string paths to Path objects for easier manipulation.
    parent_path = Path(parent_dir)
    train_path = Path(train_dir)
    test_path = Path(test_dir)

    # Create the destination train and test directories if they don't exist.
    # The `exist_ok=True` argument prevents an error if the directories already exist.
    train_path.mkdir(exist_ok=True)
    test_path.mkdir(exist_ok=True)
    
    print(f"Source directory: {parent_path.resolve()}")
    print(f"Train directory: {train_path.resolve()}")
    print(f"Test directory: {test_path.resolve()}")
    print("-" * 30)

    # --- 2. Find Subdirectories ---
    # Get a sorted list of subdirectories within the parent directory.
    # We filter to ensure we only process directories.
    subfolders = sorted([d for d in parent_path.iterdir() if d.is_dir()])

    if not subfolders:
        print(f"Error: No subdirectories found in '{parent_dir}'. Please check the path.")
        return

    # --- 3. Process Each Subfolder ---
    for subfolder in subfolders:
        subfolder_name = subfolder.name
        print(f"Processing subfolder: '{subfolder_name}'...")

        # Find all '.h5ad' files in the current subfolder.
        # We sort them to ensure a consistent split every time the script is run.
        h5ad_files = sorted(list(subfolder.glob('*.h5')))

        if not h5ad_files:
            print(f"  No .h5ad files found in '{subfolder_name}'. Skipping.")
            continue

        # --- 4. Calculate the Split ---
        num_files = len(h5ad_files)
        split_index = int(num_files * train_split_ratio)

        train_files = h5ad_files[:split_index]
        test_files = h5ad_files[split_index:]

        print(f"  Found {num_files} files. Splitting into {len(train_files)} train and {len(test_files)} test.")

        # --- 5. Copy and Rename Training Files ---
        for src_path in train_files:
            # Get the base name of the file (e.g., 'sample_000')
            base_name = src_path.stem
            # Get the file extension (e.g., '.h5ad')
            extension = src_path.suffix
            
            # Construct the new filename as per the requirement: 'sample_000_01.h5ad'
            new_filename = f"{base_name}_{subfolder_name}{extension}"
            
            # Construct the full destination path
            dest_path = train_path / new_filename
            
            # Copy the file
            shutil.copy2(src_path, dest_path)

        # --- 6. Copy and Rename Testing Files ---
        for src_path in test_files:
            base_name = src_path.stem
            extension = src_path.suffix
            new_filename = f"{base_name}_{subfolder_name}{extension}"
            dest_path = test_path / new_filename
            shutil.copy2(src_path, dest_path)
            
        print(f"  Successfully copied and renamed files from '{subfolder_name}'.")

    print("-" * 30)
    print("Script finished successfully!")


if __name__ == '__main__':
    # --- Configuration ---
    # Set the path to your main data folder.
    PARENT_DIRECTORY = "/home/can/Dropbox/Can.sacan.cs team folder/dataset/graph_samples_clip_concat/train/" #'temp/data'
    
    # Set the names for your new output folders.
    TRAIN_DIRECTORY = '/home/can/Dropbox/Can.sacan.cs team folder/dataset/split_recordings_concat/train'
    TEST_DIRECTORY = '/home/can/Dropbox/Can.sacan.cs team folder/dataset/split_recordings_concat/val'
    
    # --- Run the Script ---
    # Call the main function with your configured paths.
    split_and_rename_data(PARENT_DIRECTORY, TRAIN_DIRECTORY, TEST_DIRECTORY)
