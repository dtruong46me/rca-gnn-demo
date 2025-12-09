"""
Customer and service generator module.
Generates customers, services, and their mappings.
"""

import random
import pandas as pd
from .utils import Config


def generate_customers(config: Config) -> pd.DataFrame:
    """
    Generate customers.
    
    Args:
        config: Configuration object with customer generation parameters
        
    Returns:
        DataFrame with columns: customer_id, name, tier
    """
    customers = []
    
    for i in range(config.NUM_CUSTOMERS):
        customer = {
            'customer_id': f"C{i+1:04d}",
            'name': f"Customer_{i+1}",
            'tier': random.choice(['Gold', 'Silver', 'Bronze'])
        }
        customers.append(customer)
    
    return pd.DataFrame(customers)


def generate_services(config: Config) -> pd.DataFrame:
    """
    Generate services.
    
    Args:
        config: Configuration object with service generation parameters
        
    Returns:
        DataFrame with columns: service_id, service_type, bandwidth_mbps
    """
    services = []
    
    for i in range(config.NUM_SERVICES):
        service = {
            'service_id': f"S{i+1:04d}",
            'service_type': random.choice(config.SERVICE_TYPES),
            'bandwidth_mbps': random.choice([10, 50, 100, 500, 1000])
        }
        services.append(service)
    
    return pd.DataFrame(services)


def generate_customer_service_mapping(
    customers: pd.DataFrame,
    services: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate mapping between customers and services.
    
    Args:
        customers: DataFrame of customers
        services: DataFrame of services
        
    Returns:
        DataFrame with columns: customer_id, service_id
    """
    mappings = []
    
    # Each customer gets 1-3 services
    for _, customer in customers.iterrows():
        num_services = random.randint(1, min(3, len(services)))
        selected_services = random.sample(services['service_id'].tolist(), num_services)
        
        for service_id in selected_services:
            mappings.append({
                'customer_id': customer['customer_id'],
                'service_id': service_id
            })
    
    return pd.DataFrame(mappings)
