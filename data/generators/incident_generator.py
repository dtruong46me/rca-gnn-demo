"""
Incident generation module for sample data creation.
"""

import random
from datetime import datetime, timedelta
from collections import deque
from typing import List, Dict, Tuple
import pandas as pd

from ..config import NUM_INCIDENTS, RANDOM_SEED


def build_graph_from_edges(edges_df: pd.DataFrame, devices_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Build adjacency list graph from edges.
    
    Args:
        edges_df: DataFrame with edge information
        devices_df: DataFrame with device information
        
    Returns:
        Dictionary mapping device_id to list of connected devices
    """
    graph = {dev: [] for dev in devices_df.device_id}
    
    for _, row in edges_df.iterrows():
        graph[row["source"]].append(row["target"])
    
    return graph


def generate_incident_labels_bfs(
    root_device: str,
    graph: Dict[str, List[str]],
    all_devices: List[str]
) -> Dict[str, int]:
    """
    Generate labels for an incident using BFS from root cause.
    
    Labels:
        0 = normal (not affected)
        1 = victim (affected downstream)
        2 = root cause
    
    Args:
        root_device: Root cause device ID
        graph: Adjacency list graph
        all_devices: List of all device IDs
        
    Returns:
        Dictionary mapping device_id to label
    """
    node_label = {dev: 0 for dev in all_devices}
    node_label[root_device] = 2  # Root cause
    
    # BFS from root to mark victims
    queue = deque([root_device])
    visited = set([root_device])
    
    while queue:
        cur = queue.popleft()
        for neighbor in graph.get(cur, []):
            if neighbor not in visited:
                node_label[neighbor] = 1  # Victim
                visited.add(neighbor)
                queue.append(neighbor)
    
    return node_label


def generate_incidents_and_labels(
    devices_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    events_df: pd.DataFrame,
    num_incidents: int = NUM_INCIDENTS,
    start_time: datetime = None,
    time_range_hours: int = 24,
    min_events: int = 10,
    max_events: int = 20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate incidents with associated events and labels.
    
    Args:
        devices_df: DataFrame with device information
        edges_df: DataFrame with edge information
        events_df: DataFrame with event information
        num_incidents: Number of incidents to generate
        start_time: Start time for incidents
        time_range_hours: Time range in hours
        min_events: Minimum events per incident
        max_events: Maximum events per incident
        
    Returns:
        Tuple of (incidents_df, incident_events_df, labels_df)
    """
    random.seed(RANDOM_SEED)
    
    if start_time is None:
        start_time = datetime.now() - timedelta(days=1)
    
    # Build graph for BFS
    graph = build_graph_from_edges(edges_df, devices_df)
    all_devices = devices_df.device_id.tolist()
    
    incidents = []
    incident_events = []
    labels = []
    
    for inc in range(num_incidents):
        # Select random root cause device
        root = devices_df.sample(1).iloc[0].device_id
        ts = start_time + timedelta(seconds=random.randint(0, 3600 * time_range_hours))
        
        incident_id = f"INC_{inc}"
        
        # Create incident
        incidents.append({
            "incident_id": incident_id,
            "root_cause_device": root,
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Attach random events to incident
        num_related = random.randint(min_events, max_events)
        related = events_df.sample(min(num_related, len(events_df)))
        
        for _, r in related.iterrows():
            incident_events.append({
                "incident_id": incident_id,
                "event_id": r.event_id
            })
        
        # Generate labels using BFS
        node_labels = generate_incident_labels_bfs(root, graph, all_devices)
        
        for dev, lab in node_labels.items():
            labels.append({
                "incident_id": incident_id,
                "device_id": dev,
                "label": lab
            })
    
    return (
        pd.DataFrame(incidents),
        pd.DataFrame(incident_events),
        pd.DataFrame(labels)
    )
