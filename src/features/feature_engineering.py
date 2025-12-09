"""
Feature engineering module for RCA-GNN system.
Handles static and dynamic feature computation for devices.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple
from sklearn.preprocessing import OneHotEncoder

import networkx as nx


def fit_static_encoders(devices_df: pd.DataFrame) -> Tuple[OneHotEncoder, OneHotEncoder]:
    """
    Fit one-hot encoders for categorical device features.
    
    Args:
        devices_df: Devices DataFrame with vendor and layer columns
        
    Returns:
        Tuple of (vendor_encoder, layer_encoder)
    """
    enc_vendor = OneHotEncoder(sparse=False, handle_unknown='ignore')
    enc_layer = OneHotEncoder(sparse=False, handle_unknown='ignore')
    
    enc_vendor.fit(devices_df[['vendor']])
    enc_layer.fit(devices_df[['layer']])
    
    return enc_vendor, enc_layer


def build_static_feature_matrix(
    devices_df: pd.DataFrame,
    device_list: list,
    device_index: dict,
    enc_vendor: OneHotEncoder,
    enc_layer: OneHotEncoder
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build static feature matrices (vendor, layer) for all devices.
    
    Args:
        devices_df: Devices DataFrame
        device_list: Ordered list of device IDs
        device_index: Device ID to index mapping
        enc_vendor: Fitted vendor encoder
        enc_layer: Fitted layer encoder
        
    Returns:
        Tuple of (vendor_matrix, layer_matrix)
    """
    n = len(device_list)
    
    # Prepare arrays in index order
    vendor_vals = []
    layer_vals = []
    
    for d in device_list:
        row = devices_df[devices_df['device_id'] == d]
        if row.shape[0] == 0:
            vendor_vals.append(['UNKNOWN'])
            layer_vals.append(['UNKNOWN'])
        else:
            vendor_vals.append([row.iloc[0]['vendor']])
            layer_vals.append([row.iloc[0]['layer']])
    
    vendor_mat = enc_vendor.transform(vendor_vals)  # n x vendor_dim
    layer_mat = enc_layer.transform(layer_vals)     # n x layer_dim
    
    return vendor_mat, layer_mat


def compute_degree_features(
    G: nx.Graph,
    device_list: list,
    device_index: dict
) -> np.ndarray:
    """
    Compute node degree features from topology graph.
    
    Args:
        G: NetworkX graph of device topology
        device_list: Ordered list of device IDs
        device_index: Device ID to index mapping
        
    Returns:
        Degree feature matrix (n x 1)
    """
    deg = np.zeros((len(device_list), 1), dtype=float)
    
    for d in device_list:
        deg[device_index[d], 0] = G.degree(d)
    
    return deg


def aggregate_events_for_window(
    events_df: pd.DataFrame,
    device_list: list,
    device_index: dict,
    window_center_time: datetime,
    window_mins: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate event features for devices within a time window.
    
    Args:
        events_df: Events DataFrame with timestamp column
        device_list: Ordered list of device IDs
        device_index: Device ID to index mapping
        window_center_time: Center time of the window
        window_mins: Window size in minutes (looking back from center)
        
    Returns:
        Tuple of (event_count_matrix, critical_count_matrix)
    """
    start = window_center_time - timedelta(minutes=window_mins)
    end = window_center_time
    
    # Ensure timestamp column is datetime
    if events_df['timestamp'].dtype == object:
        events_df = events_df.copy()
        events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
    
    # Filter events in window
    subset = events_df[
        (events_df['timestamp'] >= start) & 
        (events_df['timestamp'] <= end)
    ]
    
    # Initialize count matrices
    event_count = np.zeros((len(device_list), 1), dtype=float)
    critical_count = np.zeros((len(device_list), 1), dtype=float)
    
    # Aggregate counts per device
    for _, r in subset.iterrows():
        dev = r['device_id']
        if dev in device_index:
            i = device_index[dev]
            event_count[i, 0] += 1
            if str(r.get('severity', "")).lower() == 'critical':
                critical_count[i, 0] += 1
    
    return event_count, critical_count


def build_combined_features(
    vendor_mat: np.ndarray,
    layer_mat: np.ndarray,
    degree_feat: np.ndarray,
    event_count: np.ndarray,
    critical_count: np.ndarray
) -> np.ndarray:
    """
    Combine all feature matrices into a single feature matrix.
    
    Args:
        vendor_mat: Vendor one-hot features
        layer_mat: Layer one-hot features
        degree_feat: Degree features
        event_count: Event count features
        critical_count: Critical event count features
        
    Returns:
        Combined feature matrix (n x total_dim)
    """
    return np.concatenate([
        vendor_mat,
        layer_mat,
        degree_feat,
        event_count,
        critical_count
    ], axis=1)
