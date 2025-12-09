"""
Incident and label generator module.
Generates incidents, incident-event associations, and root cause labels using BFS.
"""

import random
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
import networkx as nx
from .utils import Config


def generate_incidents(
    devices: pd.DataFrame,
    events: pd.DataFrame,
    config: Config,
    start_time: datetime
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate incidents and their associated events.
    
    Args:
        devices: DataFrame of devices
        events: DataFrame of events
        config: Configuration object with incident generation parameters
        start_time: Start time for incident generation
        
    Returns:
        Tuple of (incidents DataFrame, incident_events DataFrame)
    """
    incidents = []
    incident_events_list = []
    
    device_ids = devices['device_id'].tolist()
    events_list = events.to_dict('records')
    
    for i in range(config.NUM_INCIDENTS):
        # Random timestamp
        random_seconds = random.randint(0, config.TIME_RANGE_DAYS * 24 * 60 * 60)
        timestamp = start_time + timedelta(seconds=random_seconds)
        
        incident = {
            'incident_id': f"I{i+1:04d}",
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'description': f"Network incident {i+1}",
            'affected_device': random.choice(device_ids)
        }
        incidents.append(incident)
        
        # Associate 3-10 random events with this incident
        num_events = random.randint(3, min(10, len(events_list)))
        selected_events = random.sample(events_list, num_events)
        
        for event in selected_events:
            incident_events_list.append({
                'incident_id': incident['incident_id'],
                'event_id': event['event_id']
            })
    
    incidents_df = pd.DataFrame(incidents)
    incidents_df = incidents_df.sort_values('timestamp').reset_index(drop=True)
    incident_events_df = pd.DataFrame(incident_events_list)
    
    return incidents_df, incident_events_df


def generate_labels_bfs(
    devices: pd.DataFrame,
    edges: pd.DataFrame,
    incidents: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate root cause labels for devices using BFS from incident source.
    
    For each incident:
    - Root cause device: label = 2
    - Devices within 1-2 hops: label = 1 (victims)
    - Other devices: label = 0 (normal)
    
    Args:
        devices: DataFrame of devices
        edges: DataFrame of topology edges
        incidents: DataFrame of incidents
        
    Returns:
        DataFrame with columns: incident_id, device_id, label
    """
    labels = []
    
    # Build network graph
    G = nx.Graph()
    device_ids = devices['device_id'].tolist()
    G.add_nodes_from(device_ids)
    
    for _, edge in edges.iterrows():
        if edge['source'] in device_ids and edge['target'] in device_ids:
            G.add_edge(edge['source'], edge['target'])
    
    # For each incident, assign labels based on BFS from root cause
    for _, incident in incidents.iterrows():
        incident_id = incident['incident_id']
        root_device = incident['affected_device']
        
        # Initialize all devices as normal
        device_labels = {device_id: 0 for device_id in device_ids}
        
        # Mark root cause
        device_labels[root_device] = 2
        
        # BFS to find victims (1-2 hops away)
        visited = {root_device}
        queue = deque([(root_device, 0)])  # (device, distance)
        
        while queue:
            current_device, distance = queue.popleft()
            
            # Mark neighbors within 2 hops as victims
            if distance < 2:
                for neighbor in G.neighbors(current_device):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        device_labels[neighbor] = 1  # Victim
                        queue.append((neighbor, distance + 1))
        
        # Add labels for all devices for this incident
        for device_id, label in device_labels.items():
            labels.append({
                'incident_id': incident_id,
                'device_id': device_id,
                'label': label
            })
    
    return pd.DataFrame(labels)


def generate_incidents_and_labels(
    devices: pd.DataFrame,
    edges: pd.DataFrame,
    events: pd.DataFrame,
    config: Config,
    start_time: datetime
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate incidents, incident-event associations, and labels.
    
    This is a convenience function that combines incident and label generation.
    
    Args:
        devices: DataFrame of devices
        edges: DataFrame of topology edges
        events: DataFrame of events
        config: Configuration object
        start_time: Start time for generation
        
    Returns:
        Tuple of (incidents DataFrame, incident_events DataFrame, labels DataFrame)
    """
    incidents, incident_events = generate_incidents(devices, events, config, start_time)
    labels = generate_labels_bfs(devices, edges, incidents)
    
    return incidents, incident_events, labels
