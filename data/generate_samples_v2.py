"""
Generate sample data for RCA-GNN system (Refactored Version).

This script generates synthetic network topology, devices, events,
incidents, and labels for training and testing the RCA-GNN model.

This refactored version uses modular generators from the generators/ folder,
providing clean separation of concerns and better maintainability.

Usage:
    python generate_samples_v2.py [--output_dir OUTPUT_DIR] [--num_devices NUM] 
                                  [--num_incidents NUM] [--num_events NUM]

Examples:
    # Generate with default settings
    python generate_samples_v2.py
    
    # Specify output directory
    python generate_samples_v2.py --output_dir ./samples
    
    # Customize counts
    python generate_samples_v2.py --num_devices 100 --num_incidents 50
"""

import os
import sys
import argparse
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import from generators module
from generators import (
    generate_devices,
    generate_topology,
    generate_events,
    generate_incidents_and_labels,
    generate_customers,
    generate_services,
    generate_customer_service_mapping
)

# Import configuration
from src.config import (
    NUM_DEVICES, NUM_CUSTOMERS, NUM_SERVICES, NUM_EVENTS, NUM_INCIDENTS,
    RANDOM_SEED
)




# ============================================================================
# MAIN EXECUTION
# ============================================================================

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
        devices, edges, customers, services, events, incidents, 
        incident_events, labels: DataFrames to export
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


def generate_all_data(config: Config = Config(), output_dir: str = ".") -> None:
    """
    Main function to generate all sample data.
    
    Args:
        config: Configuration object with generation parameters
        output_dir: Directory to save generated CSV files
    """
    print("\n" + "=" * 60)
    print("RCA-GNN Sample Data Generation (Refactored v2)")
    print("=" * 60)
    
    # Initialize random seeds
    initialize_random_seeds(config.RANDOM_SEED)
    
    # Set start time
    start_time = datetime.now() - timedelta(days=config.TIME_RANGE_DAYS)
    
    # 1. Generate devices
    print("\n1. Generating devices...")
    devices = generate_devices(config)
    print(f"   ✓ Generated {len(devices)} devices")
    
    # 2. Generate topology
    print("\n2. Generating network topology...")
    edges = generate_topology(devices, config)
    print(f"   ✓ Generated {len(edges)} edges")
    
    # 3. Generate customers and services
    print("\n3. Generating customers and services...")
    customers = generate_customers(config)
    services = generate_services(config)
    customer_service = generate_customer_service_mapping(customers, services)
    print(f"   ✓ Generated {len(customers)} customers")
    print(f"   ✓ Generated {len(services)} services")
    print(f"   ✓ Generated {len(customer_service)} mappings")
    
    # 4. Generate events
    print("\n4. Generating events...")
    events = generate_events(devices, config, start_time)
    print(f"   ✓ Generated {len(events)} events")
    
    # 5. Generate incidents
    print("\n5. Generating incidents...")
    incidents, incident_events = generate_incidents(devices, events, config, start_time)
    print(f"   ✓ Generated {len(incidents)} incidents")
    print(f"   ✓ Generated {len(incident_events)} incident-event associations")
    
    # 6. Generate labels
    print("\n6. Generating labels using BFS...")
    labels = generate_labels_bfs(devices, edges, incidents)
    print(f"   ✓ Generated {len(labels)} labels")
    
    # 7. Export files
    print("\n7. Exporting data...")
    export_dataframes(
        output_dir,
        devices, edges, customers, services,
        events, incidents, incident_events, labels
    )
    
    print("=" * 60)
    print("DATA GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 60 + "\n")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate sample data for RCA-GNN system (Refactored v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate with default settings
  python generate_samples_v2.py
  
  # Specify output directory
  python generate_samples_v2.py --output_dir ./samples
  
  # Customize counts
  python generate_samples_v2.py --num_devices 100 --num_incidents 50
        """
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save generated CSV files (default: current directory)"
    )
    
    parser.add_argument(
        "--num_devices",
        type=int,
        default=Config.NUM_DEVICES,
        help=f"Number of devices to generate (default: {Config.NUM_DEVICES})"
    )
    
    parser.add_argument(
        "--num_incidents",
        type=int,
        default=Config.NUM_INCIDENTS,
        help=f"Number of incidents to generate (default: {Config.NUM_INCIDENTS})"
    )
    
    parser.add_argument(
        "--num_events",
        type=int,
        default=Config.NUM_EVENTS,
        help=f"Number of events to generate (default: {Config.NUM_EVENTS})"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=Config.RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {Config.RANDOM_SEED})"
    )
    
    args = parser.parse_args()
    
    # Create custom config if parameters provided
    config = Config()
    if args.num_devices != Config.NUM_DEVICES:
        config.NUM_DEVICES = args.num_devices
    if args.num_incidents != Config.NUM_INCIDENTS:
        config.NUM_INCIDENTS = args.num_incidents
    if args.num_events != Config.NUM_EVENTS:
        config.NUM_EVENTS = args.num_events
    if args.seed != Config.RANDOM_SEED:
        config.RANDOM_SEED = args.seed
    
    # Generate all data
    generate_all_data(config, args.output_dir)


if __name__ == "__main__":
    main()
