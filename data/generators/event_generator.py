"""
Event generator module.
Generates network events with timestamps and severity levels.
"""

import random
from datetime import datetime, timedelta
import pandas as pd
from .utils import Config


def generate_events(
    devices: pd.DataFrame,
    config: Config,
    start_time: datetime
) -> pd.DataFrame:
    """
    Generate network events for devices.
    
    Args:
        devices: DataFrame of devices
        config: Configuration object with event generation parameters
        start_time: Start time for event generation
        
    Returns:
        DataFrame with columns: event_id, device_id, event_type, severity, timestamp
    """
    events = []
    device_ids = devices['device_id'].tolist()
    
    for i in range(config.NUM_EVENTS):
        # Random timestamp within the time range
        random_seconds = random.randint(0, config.TIME_RANGE_DAYS * 24 * 60 * 60)
        timestamp = start_time + timedelta(seconds=random_seconds)
        
        event = {
            'event_id': f"E{i+1:05d}",
            'device_id': random.choice(device_ids),
            'event_type': random.choice(config.EVENT_TYPES),
            'severity': random.choice(config.SEVERITY_LIST),
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }
        events.append(event)
    
    # Sort by timestamp
    events_df = pd.DataFrame(events)
    events_df = events_df.sort_values('timestamp').reset_index(drop=True)
    
    return events_df
