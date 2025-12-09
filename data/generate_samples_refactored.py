"""
Generate sample data for RCA-GNN system.

This script generates synthetic network topology, devices, events,
incidents, and labels for training and testing the RCA-GNN model.

Usage:
    python generate_samples.py [--output_dir OUTPUT_DIR]
"""

import os
import argparse
from datetime import datetime, timedelta

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.config import (
    NUM_DEVICES, NUM_CUSTOMERS, NUM_SERVICES, NUM_EVENTS, NUM_INCIDENTS
)
from generators import (
    generate_devices,
    generate_topology,
    generate_events,
    generate_incidents_and_labels,
    generate_customers,
    generate_services,
    generate_customer_service_mapping
)


def generate_all_samples(output_dir: str = ".") -> None:
    """
    Generate all sample data files.
    
    Args:
        output_dir: Directory to save generated CSV files
    """
    print("="*60)
    print("RCA-GNN Sample Data Generation")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set start time for events/incidents
    start_time = datetime.now() - timedelta(days=1)
    
    # 1. Generate devices
    print("\n1. Generating devices...")
    df_devices = generate_devices(num_devices=NUM_DEVICES)
    print(f"   Generated {len(df_devices)} devices")
    
    # 2. Generate topology
    print("\n2. Generating network topology...")
    df_edges = generate_topology(df_devices)
    print(f"   Generated {len(df_edges)} edges")
    
    # 3. Generate customers and services
    print("\n3. Generating customers and services...")
    df_customers = generate_customers(num_customers=NUM_CUSTOMERS)
    df_services = generate_services(num_services=NUM_SERVICES)
    df_customer_service = generate_customer_service_mapping(df_customers, df_services)
    print(f"   Generated {len(df_customers)} customers")
    print(f"   Generated {len(df_services)} services")
    print(f"   Generated {len(df_customer_service)} customer-service mappings")
    
    # 4. Generate events
    print("\n4. Generating events...")
    df_events = generate_events(
        df_devices,
        num_events=NUM_EVENTS,
        start_time=start_time
    )
    print(f"   Generated {len(df_events)} events")
    
    # 5. Generate incidents and labels
    print("\n5. Generating incidents and labels...")
    df_incidents, df_incident_events, df_labels = generate_incidents_and_labels(
        df_devices,
        df_edges,
        df_events,
        num_incidents=NUM_INCIDENTS,
        start_time=start_time
    )
    print(f"   Generated {len(df_incidents)} incidents")
    print(f"   Generated {len(df_incident_events)} incident-event associations")
    print(f"   Generated {len(df_labels)} labels")
    
    # 6. Export files
    print("\n6. Exporting CSV files...")
    files = {
        "devices.csv": df_devices,
        "edges.csv": df_edges,
        "customers.csv": df_customers,
        "services.csv": df_services,
        "events.csv": df_events,
        "incidents.csv": df_incidents,
        "incident_events.csv": df_incident_events,
        "node_labels.csv": df_labels
    }
    
    for filename, df in files.items():
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"   Saved {filepath}")
    
    print("\n" + "="*60)
    print("DATA GENERATION COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"\nGenerated files in: {output_dir}")
    print("\nFiles created:")
    for filename in files.keys():
        print(f"  - {filename}")
    print("\n" + "="*60)


def main():
    """
    Main entry point with CLI argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Generate sample data for RCA-GNN system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python generate_samples.py --output_dir ./data/samples
        """
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./samples",
        help="Directory to save generated CSV files"
    )
    
    args = parser.parse_args()
    
    generate_all_samples(args.output_dir)


if __name__ == "__main__":
    main()
