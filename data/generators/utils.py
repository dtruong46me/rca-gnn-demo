"""
Utility classes and functions for data generation.
"""

import random
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration for data generation."""
    
    # Counts
    NUM_DEVICES: int = 80
    NUM_CUSTOMERS: int = 150
    NUM_SERVICES: int = 80
    NUM_EVENTS: int = 1500
    NUM_INCIDENTS: int = 30
    
    # Random seed
    RANDOM_SEED: int = 42
    
    # Time range
    TIME_RANGE_DAYS: int = 30
    
    # Device attributes
    VENDORS: list[str] = field(default_factory=lambda: ["Huawei", "ZTE", "GCOM", "Cisco", "Juniper"])
    LAYERS: list[str] = field(default_factory=lambda: ["CORE", "AGG", "ACCESS", "OLT", "ONU"])
    MODELS: list[str] = field(default_factory=lambda: ["X6000", "C300", "MA5800", "S6720", "QFX5100"])
    EVENT_TYPES: list[str] = field(default_factory=lambda: ["LOS", "linkDown", "highCPU", "highTemp", "packetDrop"])
    SEVERITY_LIST: list[str] = field(default_factory=lambda: ["critical", "major", "minor", "warning"])
    LINK_TYPES: list[str] = field(default_factory=lambda: ["fiber", "ethernet"])
    
    # Topology parameters
    MAX_EDGES_PER_NODE: int = 3
    LINK_CAPACITIES_MBPS: list[int] = field(default_factory=lambda: [100, 1000, 10000])
    
    # Service types
    SERVICE_TYPES: list[str] = field(default_factory=lambda: ["Internet", "VoIP", "VPN"])


def initialize_random_seeds(seed: int = 42) -> None:
    """
    Initialize random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
