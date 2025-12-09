"""
Export utility module.
Functions for exporting generated data to CSV files.
"""

import os
import pandas as pd


def export_dataframes(
    output_dir: str,
    devices: pd.DataFrame,
    edges: pd.DataFrame,
    customers: pd.DataFrame,
    services: pd.DataFrame,
    events: pd.DataFrame,
    incidents: pd.DataFrame,
    incident_events: pd.DataFrame,
    labels: pd.DataFrame
) -> None:
    """
    Export all DataFrames to CSV files.
    
    Args:
        output_dir: Directory to save CSV files
        devices: Devices DataFrame
        edges: Topology edges DataFrame
        customers: Customers DataFrame
        services: Services DataFrame
        events: Events DataFrame
        incidents: Incidents DataFrame
        incident_events: Incident-event associations DataFrame
        labels: Node labels DataFrame
    """
    os.makedirs(output_dir, exist_ok=True)
    
    files = {
        "devices.csv": devices,
        "edges.csv": edges,
        "customers.csv": customers,
        "services.csv": services,
        "events.csv": events,
        "incidents.csv": incidents,
        "incident_events.csv": incident_events,
        "node_labels.csv": labels
    }
    
    print(f"\nExporting files to: {output_dir}")
    print("=" * 60)
    
    for filename, df in files.items():
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"✓ Saved {filename} ({len(df)} rows)")
    
    print("=" * 60)
    print(f"Successfully generated {len(files)} files!\n")
