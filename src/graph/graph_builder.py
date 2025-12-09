"""
Graph building utilities for RCA-GNN system.
Handles topology graph construction and sample dataset creation.
"""

import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from typing import Tuple, List
from datetime import datetime

from ..features import (
    fit_static_encoders,
    build_static_feature_matrix,
    compute_degree_features,
    aggregate_events_for_window,
    build_combined_features
)


def build_topology_graph(edges_df: pd.DataFrame, device_list: list) -> nx.Graph:
    """
    Build an undirected NetworkX graph from edge data for message passing.
    
    Args:
        edges_df: DataFrame with source and target columns
        device_list: List of all device IDs
        
    Returns:
        NetworkX undirected graph
    """
    G = nx.Graph()
    G.add_nodes_from(device_list)
    
    for _, r in edges_df.iterrows():
        s = r['source']
        t = r['target']
        if s in device_list and t in device_list:
            # Add edge with all attributes except source/target
            edge_attrs = {k: r[k] for k in r.index if k not in ['source', 'target']}
            G.add_edge(s, t, **edge_attrs)
    
    return G


def build_edge_index(G: nx.Graph, device_index: dict) -> torch.Tensor:
    """
    Convert NetworkX graph to PyTorch Geometric edge index format.
    
    Args:
        G: NetworkX graph
        device_index: Device ID to index mapping
        
    Returns:
        Edge index tensor (2 x num_edges)
    """
    edge_index_list = []
    
    for u, v in G.edges():
        # Add both directions for undirected graph
        edge_index_list.append((device_index[u], device_index[v]))
        edge_index_list.append((device_index[v], device_index[u]))
    
    if len(edge_index_list) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    else:
        return torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()


def build_samples(
    devices: pd.DataFrame,
    edges: pd.DataFrame,
    events: pd.DataFrame,
    incidents: pd.DataFrame,
    labels: pd.DataFrame,
    device_list: list,
    device_index: dict,
    window_mins: int = 5
) -> Tuple[List[Data], torch.Tensor, List[str]]:
    """
    Build per-incident graph samples for training/evaluation.
    
    Args:
        devices: Devices DataFrame
        edges: Edges DataFrame
        events: Events DataFrame
        incidents: Incidents DataFrame
        labels: Labels DataFrame
        device_list: Ordered list of device IDs
        device_index: Device ID to index mapping
        window_mins: Time window size in minutes
        
    Returns:
        Tuple of (samples_list, edge_index, incident_ids)
    """
    # Build topology graph
    G = build_topology_graph(edges, device_list)
    
    # Fit encoders for static features
    enc_vendor, enc_layer = fit_static_encoders(devices)
    vendor_mat, layer_mat = build_static_feature_matrix(
        devices, device_list, device_index, enc_vendor, enc_layer
    )
    degree_feat = compute_degree_features(G, device_list, device_index)
    
    # Build edge index
    edge_index = build_edge_index(G, device_index)
    
    # Group labels by incident
    labels_grouped = labels.groupby('incident_id')
    
    samples = []
    incident_ids = []
    
    for _, inc in incidents.iterrows():
        inc_id = inc['incident_id']
        
        # Parse incident timestamp
        try:
            t = pd.to_datetime(inc['timestamp'])
        except:
            t = pd.to_datetime(datetime.now())
        
        # Aggregate event features for this time window
        ev_count, crit_count = aggregate_events_for_window(
            events, device_list, device_index, t, window_mins=window_mins
        )
        
        # Combine all features
        X = build_combined_features(
            vendor_mat, layer_mat, degree_feat, ev_count, crit_count
        )
        X = torch.tensor(X, dtype=torch.float)
        
        # Build labels for this incident
        if inc_id in labels_grouped.groups:
            lab_df = labels_grouped.get_group(inc_id)
            y = np.zeros(len(device_list), dtype=np.int64)
            
            for _, r in lab_df.iterrows():
                d = r['device_id']
                if d in device_index:
                    y[device_index[d]] = int(r['label'])
        else:
            # Skip incidents without labels
            continue
        
        y = torch.tensor(y, dtype=torch.long)
        
        # Create PyG Data object
        data = Data(x=X, edge_index=edge_index, y=y)
        samples.append(data)
        incident_ids.append(inc_id)
    
    return samples, edge_index, incident_ids
