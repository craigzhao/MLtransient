"""
Data input/output utilities.

Functions for saving and loading datasets.
"""

import pickle
import json
import numpy as np
import torch
from pathlib import Path


def save_dataset(dataset, filename, format="pickle"):
    """
    Save dataset to file.

    Args:
        dataset: Dataset dictionary
        filename: Output filename (path)
        format: "pickle", "npz", or "json" (for metadata only)
    """
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert torch tensors to numpy for saving
    dataset_to_save = {}
    for key, value in dataset.items():
        if torch.is_tensor(value):
            dataset_to_save[key] = value.numpy()
        elif isinstance(value, dict):
            # Recursively convert nested dicts
            dataset_to_save[key] = {
                k: v.numpy() if torch.is_tensor(v) else v
                for k, v in value.items()
            }
        else:
            dataset_to_save[key] = value

    if format == "pickle":
        with open(filepath, "wb") as f:
            pickle.dump(dataset_to_save, f)
        print(f"Dataset saved to {filepath}")

    elif format == "npz":
        # Separate arrays from metadata
        arrays = {}
        metadata = {}
        for key, value in dataset_to_save.items():
            if isinstance(value, np.ndarray):
                arrays[key] = value
            else:
                metadata[key] = value

        # Save arrays
        np.savez(filepath, **arrays)

        # Save metadata separately
        metadata_file = filepath.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"Dataset saved to {filepath} and {metadata_file}")

    elif format == "json":
        # Only save metadata (not arrays)
        metadata = {k: v for k, v in dataset_to_save.items()
                    if not isinstance(v, np.ndarray)}
        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Metadata saved to {filepath}")

    else:
        raise ValueError(f"Unknown format: {format}")


def load_dataset(filename, format="pickle", use_torch=False):
    """
    Load dataset from file.

    Args:
        filename: Input filename (path)
        format: "pickle" or "npz"
        use_torch: Convert arrays to PyTorch tensors

    Returns:
        Dataset dictionary
    """
    filepath = Path(filename)

    if format == "pickle":
        with open(filepath, "rb") as f:
            dataset = pickle.load(f)

    elif format == "npz":
        # Load arrays
        data = np.load(filepath)
        dataset = {key: data[key] for key in data.files}

        # Load metadata if exists
        metadata_file = filepath.with_suffix(".json")
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            dataset.update(metadata)

    else:
        raise ValueError(f"Unknown format: {format}")

    # Convert to torch if requested
    if use_torch:
        for key, value in dataset.items():
            if isinstance(value, np.ndarray):
                dataset[key] = torch.from_numpy(value).float()

    print(f"Dataset loaded from {filepath}")
    return dataset


def print_dataset_summary(dataset):
    """
    Print summary of dataset contents.

    Args:
        dataset: Dataset dictionary
    """
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)

    for key, value in dataset.items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            dtype = "torch.Tensor" if torch.is_tensor(value) else "np.ndarray"
            print(f"{key:30s}: {dtype} {value.shape}")
        elif isinstance(value, dict):
            print(f"{key:30s}: dict with {len(value)} items")
        elif isinstance(value, list):
            print(f"{key:30s}: list with {len(value)} items")
        else:
            print(f"{key:30s}: {type(value).__name__} = {value}")

    print("=" * 60 + "\n")
