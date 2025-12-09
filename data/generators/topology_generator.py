"""
Topology (edges) generation module for sample data creation.
"""

import random
from typing import List, Dict
import pandas as pd

from ..config import (
    LINK_TYPES, LINK_CAPACITIES_MBPS,
    MAX_EDGES_PER_NODE, RANDOM_SEED
)


def create_edges(
    src_df: pd.DataFrame,
    tgt_df: pd.DataFrame,
    max_edges: int = MAX_EDGES_PER_NODE,
    link_types: List[str] = LINK_TYPES,
    capacities: List[int] = LINK_CAPACITIES_MBPS
) -> List[Dict]:
    """
    Create edges between source and target device layers.
    
    Args:
        src_df: Source devices DataFrame
        tgt_df: Target devices DataFrame
        max_edges: Maximum edges per source device
        link_types: List of possible link types
        capacities: List of possible link capacities
        
    Returns:
        List of edge dictionaries
    """
    edges = []
    
    for _, src in src_df.iterrows():
        if len(tgt_df) == 0:
            continue
        
        num_targets = min(len(tgt_df), random.randint(1, max_edges))
        targets = tgt_df.sample(num_targets)
        
        for _, tgt in targets.iterrows():
            edges.append({
                "source": src.device_id,
                "target": tgt.device_id,
                "link_type": random.choice(link_types),
                "capacity_Mbps": random.choice(capacities)
            })
    
    return edges


def generate_topology(devices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate hierarchical network topology based on device layers.
    
    Args:
        devices_df: DataFrame with device information
        
    Returns:
        DataFrame with edge information
    """
    random.seed(RANDOM_SEED)
    
    # Separate devices by layer
    core = devices_df[devices_df.layer == "CORE"]
    agg = devices_df[devices_df.layer == "AGG"]
    access = devices_df[devices_df.layer == "ACCESS"]
    olt = devices_df[devices_df.layer == "OLT"]
    onu = devices_df[devices_df.layer == "ONU"]
    
    # Create hierarchical edges
    edges = []
    edges += create_edges(core, agg)
    edges += create_edges(agg, access)
    edges += create_edges(access, olt)
    edges += create_edges(olt, onu)
    
    return pd.DataFrame(edges)
