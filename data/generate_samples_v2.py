import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# -----------------------------
# 1. Generate Devices
# -----------------------------
num_devices = 80
vendors = ["Huawei", "ZTE", "GCOM", "Cisco", "Juniper"]
layers = ["CORE", "AGG", "ACCESS", "OLT", "ONU"]
models = ["X6000", "C300", "MA5800", "S6720", "QFX5100"]

devices = []
for i in range(num_devices):
    dev = {
        "device_id": f"DEV_{i}",
        "vendor": random.choice(vendors),
        "model": random.choice(models),
        "layer": random.choice(layers),
        "site": f"SITE_{random.randint(1, 40)}",
        "rack": random.randint(1, 5),
        "slot": random.randint(1, 12),
        "port": random.randint(1, 48)
    }
    devices.append(dev)

df_devices = pd.DataFrame(devices)

# -----------------------------
# 2. Generate Topology Edges
# -----------------------------
edges = []
link_types = ["fiber", "ethernet"]

# group devices by layer
core = df_devices[df_devices.layer == "CORE"]
agg = df_devices[df_devices.layer == "AGG"]
access = df_devices[df_devices.layer == "ACCESS"]
olt = df_devices[df_devices.layer == "OLT"]
onu = df_devices[df_devices.layer == "ONU"]

def create_edges(src_df, tgt_df, max_edges=3):
    edge_list = []
    for _, src in src_df.iterrows():
        targets = tgt_df.sample(min(len(tgt_df), random.randint(1, max_edges)))
        for _, tgt in targets.iterrows():
            edge_list.append({
                "source": src.device_id,
                "target": tgt.device_id,
                "link_type": random.choice(link_types),
                "capacity_Mbps": random.choice([100, 1000, 10000])
            })
    return edge_list

edges += create_edges(core, agg)
edges += create_edges(agg, access)
edges += create_edges(access, olt)
edges += create_edges(olt, onu)

df_edges = pd.DataFrame(edges)

# -----------------------------
# 3. Generate Customers + Services
# -----------------------------
num_customers = 150
customers = [{"customer_id": f"CUST_{i}", "site": f"SITE_{random.randint(1,40)}"} for i in range(num_customers)]
df_customers = pd.DataFrame(customers)

num_services = 80
services = [{"service_id": f"SRV_{i}", "type": random.choice(["Internet", "VoIP", "VPN"])} for i in range(num_services)]
df_services = pd.DataFrame(services)

# Connect customers to random services
svc_edges = []
for _, cust in df_customers.iterrows():
    srv = df_services.sample(1).iloc[0]
    svc_edges.append({
        "customer_id": cust.customer_id,
        "service_id": srv.service_id
    })
df_customer_service = pd.DataFrame(svc_edges)

# -----------------------------
# 4. Generate Events
# -----------------------------
event_types = ["LOS", "linkDown", "highCPU", "highTemp", "packetDrop"]
severity_list = ["critical", "major", "minor", "warning"]

num_events = 1500
start_time = datetime.now() - timedelta(days=1)

events = []
for i in range(num_events):
    ts = start_time + timedelta(seconds=random.randint(0, 3600*24))
    dev = df_devices.sample(1).iloc[0]
    events.append({
        "event_id": f"EV_{i}",
        "device_id": dev.device_id,
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "event_type": random.choice(event_types),
        "severity": random.choice(severity_list),
    })
df_events = pd.DataFrame(events)

# -----------------------------
# 5. Generate Incidents + propagation
# -----------------------------
num_incidents = 30
incidents = []
incident_events = []

for inc in range(num_incidents):
    root_dev = df_devices.sample(1).iloc[0].device_id
    ts = start_time + timedelta(seconds=random.randint(0, 3600*24))

    incidents.append({
        "incident_id": f"INC_{inc}",
        "root_cause_device": root_dev,
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S")
    })

    # pick 5–20 random events around timestamp as related
    related = df_events.sample(random.randint(5, 20))
    for _, r in related.iterrows():
        incident_events.append({
            "incident_id": f"INC_{inc}",
            "event_id": r.event_id
        })

df_incidents = pd.DataFrame(incidents)
df_incident_events = pd.DataFrame(incident_events)

# -----------------------------
# Export CSV files
# -----------------------------
df_devices.to_csv("devices.csv", index=False)
df_edges.to_csv("edges.csv", index=False)
df_customers.to_csv("customers.csv", index=False)
df_services.to_csv("services.csv", index=False)
df_events.to_csv("events.csv", index=False)
df_incidents.to_csv("incidents.csv", index=False)
df_incident_events.to_csv("incident_events.csv", index=False)

"/mnt/data created files: devices.csv, edges.csv, customers.csv, services.csv, events.csv, incidents.csv, incident_events.csv"
