"""
Inference utilities for RCA-GNN system.
Handles root cause prediction for new events.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datetime import datetime
from typing import List, Dict, Tuple
from sklearn.preprocessing import OneHotEncoder

from ..models import GAT_RCA
from ..features import (
    aggregate_events_for_window,
    build_static_feature_matrix,
    compute_degree_features,
    build_combined_features
)
from ..graph import build_topology_graph
from ..config import DEFAULT_TOP_K, LABEL_ROOT


def infer_from_events(
    model: GAT_RCA,
    devices_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    events_df: pd.DataFrame,
    device_list: list,
    device_index: dict,
    edge_index: torch.Tensor,
    enc_vendor: OneHotEncoder,
    enc_layer: OneHotEncoder,
    window_center_time: datetime,
    window_mins: int = 5,
    topk: int = DEFAULT_TOP_K,
    device: str = 'cpu'
) -> Tuple[List[Dict], np.ndarray]:
    """
    Perform root cause inference for a new time window of events.
    
    Args:
        model: Trained GAT_RCA model
        devices_df: Devices DataFrame
        edges_df: Edges DataFrame
        events_df: Events DataFrame with new events
        device_list: Ordered list of device IDs
        device_index: Device ID to index mapping
        edge_index: Graph edge index tensor
        enc_vendor: Fitted vendor encoder
        enc_layer: Fitted layer encoder
        window_center_time: Center time for event window
        window_mins: Window size in minutes
        topk: Number of top candidates to return
        device: Device to run inference on ('cpu' or 'cuda')
        
    Returns:
        Tuple of (top_k_candidates_list, full_probability_matrix)
    """
    # Aggregate event features for the window
    ev_count, crit_count = aggregate_events_for_window(
        events_df,
        device_list,
        device_index,
        window_center_time,
        window_mins=window_mins
    )
    
    # Build static features
    vendor_mat, layer_mat = build_static_feature_matrix(
        devices_df,
        device_list,
        device_index,
        enc_vendor,
        enc_layer
    )
    
    # Compute degree features
    G = build_topology_graph(edges_df, device_list)
    degree_feat = compute_degree_features(G, device_list, device_index)
    
    # Combine all features
    X = build_combined_features(
        vendor_mat, layer_mat, degree_feat, ev_count, crit_count
    )
    X = torch.tensor(X, dtype=torch.float).to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        logits = model(X, edge_index.to(device))  # num_nodes x 3
        probs = F.softmax(logits, dim=1).cpu().numpy()  # num_nodes x 3
    
    # Extract root cause probabilities (class 2)
    root_probs = probs[:, LABEL_ROOT]
    
    # Get top-k candidates
    top_idx = list(np.argsort(-root_probs)[:topk])
    
    results = []
    for idx in top_idx:
        results.append({
            "device_id": device_list[idx],
            "root_prob": float(root_probs[idx])
        })
    
    return results, probs


def predict_root_causes(
    model: GAT_RCA,
    features: torch.Tensor,
    edge_index: torch.Tensor,
    device_list: list,
    topk: int = DEFAULT_TOP_K,
    device: str = 'cpu'
) -> List[Dict]:
    """
    Simple prediction function given pre-computed features.
    
    Args:
        model: Trained GAT_RCA model
        features: Pre-computed node features
        edge_index: Graph edge index
        device_list: Ordered list of device IDs
        topk: Number of top candidates to return
        device: Device to run prediction on
        
    Returns:
        List of top-k root cause candidates with probabilities
    """
    model.eval()
    
    features = features.to(device)
    edge_index = edge_index.to(device)
    
    with torch.no_grad():
        logits = model(features, edge_index)
        probs = F.softmax(logits, dim=1).cpu().numpy()
    
    root_probs = probs[:, LABEL_ROOT]
    top_idx = list(np.argsort(-root_probs)[:topk])
    
    results = []
    for idx in top_idx:
        results.append({
            "device_id": device_list[idx],
            "root_prob": float(root_probs[idx]),
            "rank": len(results) + 1
        })
    
    return results
