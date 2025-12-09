"""
Customer and service generation module for sample data creation.
"""

import random
import pandas as pd
from typing import List

from ..config import (
    NUM_CUSTOMERS, NUM_SERVICES, SERVICE_TYPES, RANDOM_SEED
)


def generate_customers(
    num_customers: int = NUM_CUSTOMERS,
    num_sites: int = 40
) -> pd.DataFrame:
    """
    Generate customer data.
    
    Args:
        num_customers: Number of customers to generate
        num_sites: Number of sites
        
    Returns:
        DataFrame with customer information
    """
    random.seed(RANDOM_SEED)
    
    customers = [
        {
            "customer_id": f"CUST_{i}",
            "site": f"SITE_{random.randint(1, num_sites)}"
        }
        for i in range(num_customers)
    ]
    
    return pd.DataFrame(customers)


def generate_services(
    num_services: int = NUM_SERVICES,
    service_types: List[str] = SERVICE_TYPES
) -> pd.DataFrame:
    """
    Generate service data.
    
    Args:
        num_services: Number of services to generate
        service_types: List of possible service types
        
    Returns:
        DataFrame with service information
    """
    random.seed(RANDOM_SEED)
    
    services = [
        {
            "service_id": f"SRV_{i}",
            "type": random.choice(service_types)
        }
        for i in range(num_services)
    ]
    
    return pd.DataFrame(services)


def generate_customer_service_mapping(
    customers_df: pd.DataFrame,
    services_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate mapping between customers and services.
    
    Args:
        customers_df: DataFrame with customer information
        services_df: DataFrame with service information
        
    Returns:
        DataFrame with customer-service mappings
    """
    random.seed(RANDOM_SEED)
    
    mappings = []
    for _, c in customers_df.iterrows():
        service = services_df.sample(1).iloc[0]
        mappings.append({
            "customer_id": c.customer_id,
            "service_id": service.service_id
        })
    
    return pd.DataFrame(mappings)
