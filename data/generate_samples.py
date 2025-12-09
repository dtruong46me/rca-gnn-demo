import pandas as pd
import numpy as np
import random
from faker import Faker

fake = Faker()

# 1. Devices
device_types = ["Huawei", "ZTE", "GCOM", "Cisco", "Juniper"]
device_roles = ["OLT", "Switch", "Router", "Core Router", "Aggregation Switch"]

num_devices = 80
devices = []
for i in range(num_devices):
    devices.append({
        "device_id": f"D{i+1}",
        "device_name": fake.hostname(),
        "vendor": random.choice(device_types),
        "role": random.choice(device_roles),
        "site_id": f"S{random.randint(1,40)}"
    })
df_devices = pd.DataFrame(devices)
df_devices.to_csv("devices.csv", index=False)

# 2. Sites
num_sites = 40
sites = []
for i in range(num_sites):
    sites.append({
        "site_id": f"S{i+1}",
        "site_name": fake.city(),
        "address": fake.address()
    })
df_sites = pd.DataFrame(sites)
df_sites.to_csv("sites.csv", index=False)

# 3. Customers
num_customers = 180
customers = []
for i in range(num_customers):
    customers.append({
        "customer_id": f"CUST{i+1}",
        "name": fake.name(),
        "address": fake.address(),
    })
df_customers = pd.DataFrame(customers)
df_customers.to_csv("customers.csv", index=False)

# 4. Services
num_services = 70
services = []
for i in range(num_services):
    services.append({
        "service_id": f"SV{i+1}",
        "customer_id": f"CUST{random.randint(1,num_customers)}",
        "device_id": f"D{random.randint(1,num_devices)}",
        "service_type": random.choice(["Internet", "TV", "VoIP", "VPN"])
    })
df_services = pd.DataFrame(services)
df_services.to_csv("services.csv", index=False)

# 5. Events (raw logs)
num_events = 1500
events = []
event_types = ["LOS", "High CPU", "Flap", "Link Down", "Packet Drop", "Temperature High"]
for i in range(num_events):
    events.append({
        "event_id": f"E{i+1}",
        "timestamp": fake.date_time_this_year().isoformat(),
        "device_id": f"D{random.randint(1,num_devices)}",
        "severity": random.choice(["critical", "major", "minor"]),
        "event_type": random.choice(event_types),
        "raw_log": fake.text(120)
    })
df_events = pd.DataFrame(events)
df_events.to_csv("events.csv", index=False)

# 6. Incidents
num_incidents = 35
incidents = []
root_causes = ["Fiber Cut", "Power Loss", "Device Failure", "Misconfiguration", "Traffic Congestion"]
for i in range(num_incidents):
    rc = random.choice(root_causes)
    incidents.append({
        "incident_id": f"INC{i+1}",
        "start_time": fake.date_time_this_year().isoformat(),
        "end_time": fake.date_time_this_year().isoformat(),
        "affected_device": f"D{random.randint(1,num_devices)}",
        "root_cause": rc,
        "description": f"Service impact caused by {rc.lower()}"
    })
df_incidents = pd.DataFrame(incidents)
df_incidents.to_csv("incidents.csv", index=False)
