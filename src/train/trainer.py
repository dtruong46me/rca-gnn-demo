"""
Training and evaluation utilities for RCA-GNN system.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Any

from ..models import GAT_RCA
from ..config import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_EPOCHS,
    DEFAULT_TEST_SIZE,
    RANDOM_SEED,
    LABEL_ROOT
)


def evaluate_model(
    model: GAT_RCA,
    samples: List,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Evaluate model performance on a set of samples.
    
    Metrics computed:
        - Top-1 accuracy for root cause prediction
        - Top-3 accuracy for root cause prediction
        - Per-class accuracy for all labels
    
    Args:
        model: Trained GAT_RCA model
        samples: List of PyG Data objects
        device: Device to run evaluation on ('cpu' or 'cuda')
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    total = len(samples)
    correct_root_top1 = 0
    top3_hit = 0
    perclass_acc = {0: 0, 1: 0, 2: 0}
    counts = {0: 0, 1: 0, 2: 0}
    
    with torch.no_grad():
        for data in samples:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            y = data.y.to(device)  # num_nodes
            
            # Forward pass
            logits = model(x, edge_index)  # num_nodes x 3
            probs = F.softmax(logits, dim=1).cpu().numpy()  # num_nodes x 3
            
            # Root cause prediction (class 2)
            root_scores = probs[:, LABEL_ROOT]
            pred_root_idx = int(root_scores.argmax())
            
            # Find true root (assuming single root per incident)
            true_roots = (y.cpu().numpy() == LABEL_ROOT).nonzero()[0]
            true_root_idx = int(true_roots[0]) if len(true_roots) > 0 else -1
            
            # Top-1 accuracy
            if pred_root_idx == true_root_idx:
                correct_root_top1 += 1
            
            # Top-3 accuracy
            top3 = list(np.argsort(-root_scores)[:3])
            if true_root_idx in top3:
                top3_hit += 1
            
            # Per-class accuracy
            pred_labels = logits.argmax(dim=1).cpu().numpy()
            y_np = y.cpu().numpy()
            
            for c in [0, 1, 2]:
                counts[c] += (y_np == c).sum()
                perclass_acc[c] += ((pred_labels == c) & (y_np == c)).sum()
    
    # Compute final rates
    top1_rate = correct_root_top1 / max(1, total)
    top3_rate = top3_hit / max(1, total)
    perclass_acc_final = {
        c: (perclass_acc[c] / counts[c] if counts[c] > 0 else None)
        for c in perclass_acc
    }
    
    return {
        "top1": top1_rate,
        "top3": top3_rate,
        "perclass_acc": perclass_acc_final,
        "num_samples": total
    }


def train_model(
    samples: List,
    in_dim: int,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LEARNING_RATE,
    device: str = 'cpu'
) -> Tuple[GAT_RCA, Dict[str, Any]]:
    """
    Train GAT-RCA model on incident samples.
    
    Args:
        samples: List of PyG Data objects
        in_dim: Input feature dimension
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on ('cpu' or 'cuda')
        
    Returns:
        Tuple of (trained_model, training_info_dict)
    """
    # Train/test split
    train_idx, test_idx = train_test_split(
        list(range(len(samples))),
        test_size=DEFAULT_TEST_SIZE,
        random_state=RANDOM_SEED
    )
    train_samples = [samples[i] for i in train_idx]
    test_samples = [samples[i] for i in test_idx]
    
    # Initialize model and optimizer
    model = GAT_RCA(in_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=DEFAULT_WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()  # Per-node classification
    
    # Training loop
    best_top1 = -1.0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for data in train_samples:
            optimizer.zero_grad()
            
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            y = data.y.to(device)  # num_nodes
            
            # Forward pass
            logits = model(x, edge_index)  # num_nodes x 3
            loss = criterion(logits, y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Compute average loss
        avg_loss = total_loss / max(1, len(train_samples))
        
        # Evaluate on test set
        stats = evaluate_model(model, test_samples, device=device)
        
        print(
            f"Epoch {epoch:02d} "
            f"loss={avg_loss:.4f} "
            f"top1={stats['top1']:.3f} "
            f"top3={stats['top3']:.3f}"
        )
        
        # Save best model
        if stats['top1'] > best_top1:
            best_top1 = stats['top1']
            best_state = model.state_dict()
    
    # Load best model state
    if best_state is not None:
        model.load_state_dict(best_state)
    
    training_info = {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "metrics": stats
    }
    
    return model, training_info
