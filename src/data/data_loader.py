"""
Data loading utilities for RCA-GNN system.
Handles CSV file loading and basic preprocessing.
"""

import os
import pandas as pd
from typing import Tuple

from ..config import (
    CSV_DEVICES, CSV_EDGES, CSV_EVENTS, 
    CSV_INCIDENTS, CSV_LABELS, CSV_CUSTOMERS, CSV_SERVICES
)


def load_csvs(data_dir: str = ".") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all required CSV files for training/inference.
    
    Args:
        data_dir: Directory containing the CSV files
        
    Returns:
        Tuple of (devices, edges, events, incidents, labels) DataFrames
    """
    devices = pd.read_csv(os.path.join(data_dir, CSV_DEVICES))
    edges = pd.read_csv(os.path.join(data_dir, CSV_EDGES))
    events = pd.read_csv(os.path.join(data_dir, CSV_EVENTS))
    incidents = pd.read_csv(os.path.join(data_dir, CSV_INCIDENTS))
    labels = pd.read_csv(os.path.join(data_dir, CSV_LABELS))
    
    return devices, edges, events, incidents, labels


def load_full_csvs(data_dir: str = ".") -> dict:
    """
    Load all CSV files including customers and services.
    
    Args:
        data_dir: Directory containing the CSV files
        
    Returns:
        Dictionary with all DataFrames
    """
    return {
        'devices': pd.read_csv(os.path.join(data_dir, CSV_DEVICES)),
        'edges': pd.read_csv(os.path.join(data_dir, CSV_EDGES)),
        'events': pd.read_csv(os.path.join(data_dir, CSV_EVENTS)),
        'incidents': pd.read_csv(os.path.join(data_dir, CSV_INCIDENTS)),
        'labels': pd.read_csv(os.path.join(data_dir, CSV_LABELS)),
        'customers': pd.read_csv(os.path.join(data_dir, CSV_CUSTOMERS)),
        'services': pd.read_csv(os.path.join(data_dir, CSV_SERVICES))
    }


def preprocess_timestamps(events_df: pd.DataFrame, incidents_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure timestamp columns are parsed as datetime objects.
    
    Args:
        events_df: Events DataFrame
        incidents_df: Incidents DataFrame
        
    Returns:
        Tuple of processed (events, incidents) DataFrames
    """
    events_df = events_df.copy()
    incidents_df = incidents_df.copy()
    
    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
    incidents_df['timestamp'] = pd.to_datetime(incidents_df['timestamp'])
    
    return events_df, incidents_df


def build_device_index(devices_df: pd.DataFrame) -> Tuple[list, dict]:
    """
    Create ordered device list and index mapping.
    
    Args:
        devices_df: Devices DataFrame
        
    Returns:
        Tuple of (device_list, device_index_dict)
    """
    device_list = devices_df['device_id'].unique().tolist()
    device_list.sort()
    device_index = {d: i for i, d in enumerate(device_list)}
    
    return device_list, device_index
