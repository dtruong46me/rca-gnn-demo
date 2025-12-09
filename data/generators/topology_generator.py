"""
Topology generator module.
Generates network topology edges between devices.
"""

import random
import pandas as pd
from .utils import Config


def generate_topology(devices: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Generate network topology as edges connecting devices.
    
    Args:
        devices: DataFrame of devices
        config: Configuration object with topology parameters
        
    Returns:
        DataFrame with columns: source, target, link_type, capacity_mbps
    """
    edges = []
    device_ids = devices['device_id'].tolist()
    
    # Create a connected network with some randomness
    for i, device_id in enumerate(device_ids):
        # Connect to a few random other devices
        num_connections = random.randint(1, min(config.MAX_EDGES_PER_NODE, len(device_ids) - 1))
        
        # Prefer connecting to nearby devices (by index) but allow some random connections
        candidates = device_ids[:i] + device_ids[i+1:]
        if not candidates:
            continue
            
        targets = random.sample(candidates, min(num_connections, len(candidates)))
        
        for target in targets:
            # Avoid duplicate edges
            if not any(e['source'] == target and e['target'] == device_id for e in edges):
                edge = {
                    'source': device_id,
                    'target': target,
                    'link_type': random.choice(config.LINK_TYPES),
                    'capacity_mbps': random.choice(config.LINK_CAPACITIES_MBPS)
                }
                edges.append(edge)
    
    return pd.DataFrame(edges)
