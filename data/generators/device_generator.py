"""
Device generator module.
Generates network devices with various attributes.
"""

import random
import pandas as pd
from .utils import Config


def generate_devices(config: Config) -> pd.DataFrame:
    """
    Generate network devices with various attributes.
    
    Args:
        config: Configuration object with device generation parameters
        
    Returns:
        DataFrame with columns: device_id, vendor, layer, model, location
    """
    devices = []
    
    for i in range(config.NUM_DEVICES):
        device = {
            'device_id': f"D{i+1:04d}",
            'vendor': random.choice(config.VENDORS),
            'layer': random.choice(config.LAYERS),
            'model': random.choice(config.MODELS),
            'location': f"Site_{random.randint(1, 20)}"
        }
        devices.append(device)
    
    return pd.DataFrame(devices)
