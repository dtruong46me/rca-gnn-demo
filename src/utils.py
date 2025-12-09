"""
Utility functions for RCA-GNN system.
Common helper functions used across modules.
"""

import os
import json
import torch
from typing import Dict, Any

from .config import MODEL_STATE_FILE, META_FILE


def save_model_and_metadata(
    model: torch.nn.Module,
    device_list: list,
    device_index: dict,
    output_dir: str
) -> None:
    """
    Save trained model state and metadata to disk.
    
    Args:
        model: Trained PyTorch model
        device_list: Ordered list of device IDs
        device_index: Device ID to index mapping
        output_dir: Directory to save files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model state
    model_path = os.path.join(output_dir, MODEL_STATE_FILE)
    torch.save(model.state_dict(), model_path)
    
    # Save metadata
    meta = {
        "device_list": device_list,
        "device_index": device_index
    }
    meta_path = os.path.join(output_dir, META_FILE)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"Saved model and metadata to {output_dir}")


def load_metadata(output_dir: str) -> Dict[str, Any]:
    """
    Load metadata from disk.
    
    Args:
        output_dir: Directory containing metadata file
        
    Returns:
        Dictionary with device_list and device_index
    """
    meta_path = os.path.join(output_dir, META_FILE)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return meta


def load_model_state(
    model: torch.nn.Module,
    output_dir: str,
    device: str = 'cpu'
) -> torch.nn.Module:
    """
    Load model state from disk.
    
    Args:
        model: Initialized model (architecture must match saved state)
        output_dir: Directory containing model file
        device: Device to load model onto
        
    Returns:
        Model with loaded state
    """
    model_path = os.path.join(output_dir, MODEL_STATE_FILE)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def get_device() -> str:
    """
    Get the best available device (cuda if available, otherwise cpu).
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def print_model_summary(model: torch.nn.Module, device: str) -> None:
    """
    Print a summary of the model architecture and device.
    
    Args:
        model: PyTorch model
        device: Device model is on
    """
    print("\n" + "="*60)
    print("Model Architecture:")
    print("="*60)
    print(model)
    print(f"\nTraining device: {device}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print("="*60 + "\n")
