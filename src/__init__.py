"""
RCA-GNN: Root Cause Analysis using Graph Neural Networks

A modular system for analyzing network incidents and identifying
root causes using Graph Attention Networks.
"""

__version__ = "2.0.0"
__author__ = "RCA-GNN Team"

# Import main components for easy access
from .config import *
from .models import GAT_RCA
from .data import load_csvs, build_device_index
from .train import train_model, evaluate_model
from .inference import infer_from_events, predict_root_causes

__all__ = [
    'GAT_RCA',
    'load_csvs',
    'build_device_index',
    'train_model',
    'evaluate_model',
    'infer_from_events',
    'predict_root_causes'
]
