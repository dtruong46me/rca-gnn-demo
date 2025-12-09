"""
RCA-GNN: Root Cause Analysis using Graph Attention Networks

Main entry point for training and inference.
Orchestrates all modules for end-to-end workflow.
"""

import os
import argparse
import pandas as pd
import torch

from config import (
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WINDOW_MINUTES,
    DEFAULT_TOP_K,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR
)
from data import load_csvs, preprocess_timestamps, build_device_index
from features import fit_static_encoders
from graph import build_samples
from models import GAT_RCA
from train import train_model
from inference import infer_from_events
from utils import (
    save_model_and_metadata,
    load_metadata,
    load_model_state,
    get_device,
    print_model_summary
)


def train_mode(args):
    """
    Training mode: Build samples, train model, and save results.
    
    Args:
        args: Command-line arguments
    """
    print("\n" + "="*60)
    print("TRAINING MODE")
    print("="*60)
    
    # Load data
    print("\nLoading CSV files...")
    devices, edges, events, incidents, labels = load_csvs(args.data_dir)
    events, incidents = preprocess_timestamps(events, incidents)
    
    # Build device index
    device_list, device_index = build_device_index(devices)
    print(f"Loaded {len(device_list)} devices")
    
    # Build samples
    print(f"\nBuilding samples (window={args.window_mins} mins)...")
    samples, edge_index, incident_ids = build_samples(
        devices, edges, events, incidents, labels,
        device_list, device_index,
        window_mins=args.window_mins
    )
    print(f"Built {len(samples)} samples from {len(incident_ids)} incidents")
    
    # Get input dimension
    in_dim = samples[0].x.shape[1]
    print(f"Input feature dimension: {in_dim}")
    
    # Get device
    device = get_device()
    
    # Initialize and train model
    print("\nInitializing model...")
    model = GAT_RCA(in_dim)
    print_model_summary(model, device)
    
    print("Starting training...")
    model, info = train_model(
        samples,
        in_dim,
        epochs=args.epochs,
        lr=args.lr,
        device=device
    )
    
    # Print final results
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Test set metrics:")
    print(f"  Top-1 Accuracy: {info['metrics']['top1']:.4f}")
    print(f"  Top-3 Accuracy: {info['metrics']['top3']:.4f}")
    print(f"  Per-class Accuracy: {info['metrics']['perclass_acc']}")
    print(f"  Number of test samples: {info['metrics']['num_samples']}")
    
    # Save model and metadata
    print(f"\nSaving model to {args.out_dir}...")
    save_model_and_metadata(model, device_list, device_index, args.out_dir)
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60 + "\n")


def infer_mode(args):
    """
    Inference mode: Load model and predict root causes for new events.
    
    Args:
        args: Command-line arguments
    """
    print("\n" + "="*60)
    print("INFERENCE MODE")
    print("="*60)
    
    # Load data
    print("\nLoading CSV files...")
    devices, edges, events, incidents, labels = load_csvs(args.data_dir)
    events, incidents = preprocess_timestamps(events, incidents)
    
    # Build device index
    device_list, device_index = build_device_index(devices)
    print(f"Loaded {len(device_list)} devices")
    
    # Build samples to get edge_index and in_dim
    print("\nBuilding samples for structure...")
    samples, edge_index, _ = build_samples(
        devices, edges, events, incidents, labels,
        device_list, device_index,
        window_mins=args.window_mins
    )
    in_dim = samples[0].x.shape[1]
    
    # Load model
    print(f"\nLoading model from {args.out_dir}...")
    device = get_device()
    model = GAT_RCA(in_dim).to(device)
    model = load_model_state(model, args.out_dir, device=device)
    print("Model loaded successfully")
    
    # Prepare encoders
    enc_vendor, enc_layer = fit_static_encoders(devices)
    
    # Determine inference time
    if args.infer_time:
        t = pd.to_datetime(args.infer_time)
        print(f"\nInference time: {t}")
    else:
        t = pd.to_datetime("now")
        print(f"\nInference time: now ({t})")
    
    # Run inference
    print(f"\nRunning inference (window={args.window_mins} mins, top-k={args.topk})...")
    results, probs = infer_from_events(
        model, devices, edges, events,
        device_list, device_index, edge_index,
        enc_vendor, enc_layer,
        t,
        window_mins=args.window_mins,
        topk=args.topk,
        device=device
    )
    
    # Print results
    print("\n" + "="*60)
    print("ROOT CAUSE PREDICTION RESULTS")
    print("="*60)
    print(f"\nTop-{args.topk} Root Cause Candidates:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. {result['device_id']}: {result['root_prob']:.4f}")
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETED")
    print("="*60 + "\n")


def main():
    """
    Main entry point with CLI argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="RCA-GNN: Root Cause Analysis using Graph Attention Networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train model:
    python main.py --mode train --data_dir ./data/samples --epochs 40
    
  Run inference:
    python main.py --mode infer --data_dir ./data/samples --topk 5
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "infer"],
        help="Operation mode: train or infer"
    )
    
    # Data parameters
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Directory containing CSV files"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save/load model and metadata"
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate"
    )
    
    # Feature parameters
    parser.add_argument(
        "--window_mins",
        type=int,
        default=DEFAULT_WINDOW_MINUTES,
        help="Time window size in minutes for event aggregation"
    )
    
    # Inference parameters
    parser.add_argument(
        "--infer_time",
        type=str,
        default=None,
        help="ISO timestamp for inference (optional, defaults to now)"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of top root cause candidates to return"
    )
    
    args = parser.parse_args()
    
    # Route to appropriate mode
    if args.mode == 'train':
        train_mode(args)
    elif args.mode == 'infer':
        infer_mode(args)
    else:
        print(f"Unknown mode: {args.mode}")
        print("Use --mode train or --mode infer")


if __name__ == "__main__":
    main()
