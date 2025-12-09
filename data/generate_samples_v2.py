import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from collections import deque

# -----------------------------------------
# CONFIG
# -----------------------------------------
random.seed(42)
np.random.seed(42)

NUM_DEVICES = 80
NUM_CUSTOMERS = 150
NUM_SERVICES = 80
NUM_EVENTS = 1500
NUM_INCIDENTS = 30

vendors = ["Huawei", "ZTE", "GCOM", "Cisco", "Juniper"]
layers = ["CORE", "AGG", "ACCESS", "OLT", "ONU"]
models = ["X6000", "C300", "MA5800", "S6720", "QFX5100"]
event_types = ["LOS", "linkDown", "highCPU", "highTemp", "packetDrop"]
severity_list = ["critical", "major", "minor", "warning"]
link_types = ["fiber", "ethernet"]

start_time = datetime.now() - timedelta(days=1)

# -----------------------------------------
# 1. Generate Devices
# -----------------------------------------
devices = []
for i in range(NUM_DEVICES):
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

# -----------------------------------------
# 2. Generate Topology Edges
# -----------------------------------------
core = df_devices[df_devices.layer == "CORE"]
agg = df_devices[df_devices.layer == "AGG"]
access = df_devices[df_devices.layer == "ACCESS"]
olt = df_devices[df_devices.layer == "OLT"]
onu = df_devices[df_devices.layer == "ONU"]

def create_edges(src_df, tgt_df, max_edges=3):
    edges = []
    for _, src in src_df.iterrows():
        if len(tgt_df) == 0:
            continue
        targets = tgt_df.sample(min(len(tgt_df), random.randint(1, max_edges)))
        for _, tgt in targets.iterrows():
            edges.append({
                "source": src.device_id,
                "target": tgt.device_id,
                "link_type": random.choice(link_types),
                "capacity_Mbps": random.choice([100, 1000, 10000])
            })
    return edges

edges = []
edges += create_edges(core, agg)
edges += create_edges(agg, access)
edges += create_edges(access, olt)
edges += create_edges(olt, onu)

df_edges = pd.DataFrame(edges)

# -----------------------------------------
# 3. Customers + Services
# -----------------------------------------
df_customers = pd.DataFrame([
    {"customer_id": f"CUST_{i}", "site": f"SITE_{random.randint(1,40)}"}
    for i in range(NUM_CUSTOMERS)
])

df_services = pd.DataFrame([
    {"service_id": f"SRV_{i}", "type": random.choice(["Internet", "VoIP", "VPN"])}
    for i in range(NUM_SERVICES)
])

df_customer_service = pd.DataFrame([
    {"customer_id": c.customer_id, "service_id": df_services.sample(1).iloc[0].service_id}
    for _, c in df_customers.iterrows()
])

# -----------------------------------------
# 4. Events
# -----------------------------------------
events = []
for i in range(NUM_EVENTS):
    ts = start_time + timedelta(seconds=random.randint(0, 3600*24))
    dev = df_devices.sample(1).iloc[0]
    events.append({
        "event_id": f"EV_{i}",
        "device_id": dev.device_id,
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "event_type": random.choice(event_types),
        "severity": random.choice(severity_list)
    })

df_events = pd.DataFrame(events)

# -----------------------------------------
# 5. Generate Incidents
# -----------------------------------------
incidents = []
incident_events = []

for inc in range(NUM_INCIDENTS):
    root = df_devices.sample(1).iloc[0].device_id
    ts = start_time + timedelta(seconds=random.randint(0, 3600*24))

    incidents.append({
        "incident_id": f"INC_{inc}",
        "root_cause_device": root,
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S")
    })

    # randomly attach some events
    related = df_events.sample(random.randint(10, 20))
    for _, r in related.iterrows():
        incident_events.append({
            "incident_id": f"INC_{inc}",
            "event_id": r.event_id
        })

df_incidents = pd.DataFrame(incidents)
df_incident_events = pd.DataFrame(incident_events)

# -----------------------------------------
# 6. Build Graph for BFS
# -----------------------------------------
graph = {}
for dev in df_devices.device_id:
    graph[dev] = []

for _, row in df_edges.iterrows():
    graph[row["source"]].append(row["target"])  # directional

# -----------------------------------------
# 7. LABEL GENERATION (0,1,2)
# -----------------------------------------
labels = []

for _, inc in df_incidents.iterrows():
    root = inc.root_cause_device
    incident_id = inc.incident_id

    node_label = {dev: 0 for dev in df_devices.device_id}  # default 0
    node_label[root] = 2  # root cause = 2

    # BFS from root → mark victim nodes
    queue = deque([root])
    visited = set([root])

    while queue:
        cur = queue.popleft()
        for neighbor in graph.get(cur, []):
            if neighbor not in visited:
                node_label[neighbor] = 1  # victim
                visited.add(neighbor)
                queue.append(neighbor)

    # save label rows
    for dev, lab in node_label.items():
        labels.append({
            "incident_id": incident_id,
            "device_id": dev,
            "label": lab
        })

df_labels = pd.DataFrame(labels)

# -----------------------------------------
# EXPORT FILES
# -----------------------------------------
df_devices.to_csv("devices.csv", index=False)
df_edges.to_csv("edges.csv", index=False)
df_customers.to_csv("customers.csv", index=False)
df_services.to_csv("services.csv", index=False)
df_events.to_csv("events.csv", index=False)
df_incidents.to_csv("incidents.csv", index=False)
df_incident_events.to_csv("incident_events.csv", index=False)
df_labels.to_csv("node_labels.csv", index=False)

print("Generated files:")
print("devices.csv, edges.csv, customers.csv, services.csv, events.csv, incidents.csv, incident_events.csv, node_labels.csv")
