"""
Device generation module for sample data creation.
"""

import random
from typing import List, Dict
import pandas as pd

from ..config import (
    NUM_DEVICES, VENDORS, MODELS, LAYERS,
    RANDOM_SEED
)


def generate_devices(
    num_devices: int = NUM_DEVICES,
    vendors: List[str] = VENDORS,
    models: List[str] = MODELS,
    layers: List[str] = LAYERS,
    num_sites: int = 40,
    max_rack: int = 5,
    max_slot: int = 12,
    max_port: int = 48
) -> pd.DataFrame:
    """
    Generate random device data.
    
    Args:
        num_devices: Number of devices to generate
        vendors: List of possible vendors
        models: List of possible models
        layers: List of possible network layers
        num_sites: Number of sites
        max_rack: Maximum rack number
        max_slot: Maximum slot number
        max_port: Maximum port number
        
    Returns:
        DataFrame with device information
    """
    random.seed(RANDOM_SEED)
    
    devices = []
    for i in range(num_devices):
        dev = {
            "device_id": f"DEV_{i}",
            "vendor": random.choice(vendors),
            "model": random.choice(models),
            "layer": random.choice(layers),
            "site": f"SITE_{random.randint(1, num_sites)}",
            "rack": random.randint(1, max_rack),
            "slot": random.randint(1, max_slot),
            "port": random.randint(1, max_port)
        }
        devices.append(dev)
    
    return pd.DataFrame(devices)
