"""
Event generation module for sample data creation.
"""

import random
from datetime import datetime, timedelta
from typing import List
import pandas as pd

from ..config import (
    NUM_EVENTS, EVENT_TYPES, SEVERITY_LIST, RANDOM_SEED
)


def generate_events(
    devices_df: pd.DataFrame,
    num_events: int = NUM_EVENTS,
    event_types: List[str] = EVENT_TYPES,
    severities: List[str] = SEVERITY_LIST,
    start_time: datetime = None,
    time_range_hours: int = 24
) -> pd.DataFrame:
    """
    Generate random event data for devices.
    
    Args:
        devices_df: DataFrame with device information
        num_events: Number of events to generate
        event_types: List of possible event types
        severities: List of possible severity levels
        start_time: Start time for events (defaults to 1 day ago)
        time_range_hours: Time range in hours for event distribution
        
    Returns:
        DataFrame with event information
    """
    random.seed(RANDOM_SEED)
    
    if start_time is None:
        start_time = datetime.now() - timedelta(days=1)
    
    events = []
    for i in range(num_events):
        ts = start_time + timedelta(seconds=random.randint(0, 3600 * time_range_hours))
        dev = devices_df.sample(1).iloc[0]
        
        events.append({
            "event_id": f"EV_{i}",
            "device_id": dev.device_id,
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": random.choice(event_types),
            "severity": random.choice(severities)
        })
    
    return pd.DataFrame(events)
