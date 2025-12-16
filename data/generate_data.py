# generate_data.py
import pandas as pd
import networkx as nx
import random
import uuid
from datetime import datetime, timedelta
import numpy as np

# ==========================================
# CẤU HÌNH DỮ LIỆU LỚN
# ==========================================
CONFIG = {
    "NUM_CORE": 4,
    "NUM_AGG": 20,
    "NUM_ACCESS": 100,
    "NUM_ONT": 3000,          
    "START_TIME": datetime(2025, 12, 1, 0, 0, 0),
    "DAYS": 30,               
    "INCIDENTS_PER_DAY": 10,   
    "NOISE_PER_DAY": 300,     
    "PROPAGATION_RATE": 0.7,  
}

VENDORS = {
    "CORE": ["Juniper MX960", "Cisco ASR9000"],
    "AGG": ["Cisco Nexus", "H3C S12500", "Huawei S9300"],
    "ACCESS": ["Huawei MA5600T", "ZTE C320", "GCOM"],
    "ONT": ["Huawei HG8145", "ZTE F670", "Dasan"]
}

ERROR_TEMPLATES = [
    "OSPF State Change to Down", "BGP Neighbor Down", "Interface Down",
    "High Optical Loss (-35dBm)", "ERROR_Lost 100%", "CPU Overload 99%",
    "Power Supply Failure", "Fan Status Fault", "Line Protocol Down"
]
NOISE_TEMPLATES = [
    "Bảo trì có kế hoạch", "Cấu hình VLAN", "Kiểm tra tín hiệu", 
    "Khách hàng báo chậm", "Thay đổi gói cước", "Reset port theo yêu cầu",
    "Upgrade Firmware", "Thay đổi mật khẩu wifi", "Kiểm tra định tuyến"
]

def build_topology():
    G = nx.DiGraph()
    devices = []
    
    # Tạo Core
    cores = [f"CORE-{i:02d}" for i in range(CONFIG["NUM_CORE"])]
    for d in cores: 
        devices.append({"id": d, "type": "CORE", "vendor": random.choice(VENDORS["CORE"])})
    
    # Tạo Agg
    aggs = [f"AGG-{i:03d}" for i in range(CONFIG["NUM_AGG"])]
    for d in aggs: 
        devices.append({"id": d, "type": "AGG", "vendor": random.choice(VENDORS["AGG"])})
        # Nối lên Core
        parent = random.choice(cores)
        G.add_edge(parent, d)
        
    # Tạo Access
    accesses = [f"ACCESS-{i:04d}" for i in range(CONFIG["NUM_ACCESS"])]
    for d in accesses:
        devices.append({"id": d, "type": "ACCESS", "vendor": random.choice(VENDORS["ACCESS"])})
        # Nối lên Agg
        parent = random.choice(aggs)
        G.add_edge(parent, d)
        
    # Tạo ONT
    onts = [f"ONT-{i:05d}" for i in range(CONFIG["NUM_ONT"])]
    for d in onts:
        devices.append({"id": d, "type": "ONT", "vendor": random.choice(VENDORS["ONT"])})
        # Nối lên Access
        parent = random.choice(accesses)
        G.add_edge(parent, d)

    return G, devices

def simulate_traffic(G, devices):
    tickets = []
    current_time = CONFIG["START_TIME"]
    end_time = current_time + timedelta(days=CONFIG["DAYS"])
    
    # Chỉ thiết bị mạng mới hay làm Root Cause, ONT thường là nạn nhân
    root_nodes = [d["id"] for d in devices if d["type"] in ["AGG", "ACCESS", "CORE"]]
    all_node_ids = [d["id"] for d in devices]

    while current_time < end_time:
        # 1. Sinh sự cố (Incidents)
        # Poisson distribution để số lượng sự cố biến thiên tự nhiên
        daily_incidents = np.random.poisson(CONFIG["INCIDENTS_PER_DAY"])
        
        for _ in range(daily_incidents):
            root_id = random.choice(root_nodes)
            # Thời điểm xảy ra lỗi trong ngày
            incident_time = current_time + timedelta(minutes=random.randint(0, 1400))
            
            # Ticket cho Root Cause (Thủ phạm)
            tickets.append({
                "Ticket_ID": uuid.uuid4().hex[:8],
                "Device_ID": root_id,
                "Timestamp": incident_time,
                "Description": random.choice(ERROR_TEMPLATES),
                "Is_Root_Cause": 1
            })
            
            # Lan truyền lỗi xuống con cháu (Nạn nhân)
            try:
                descendants = list(nx.descendants(G, root_id))
            except:
                descendants = []

            for child in descendants:
                if random.random() < CONFIG["PROPAGATION_RATE"]:
                    delay = random.randint(1, 300) # Trễ 1-5 phút
                    tickets.append({
                        "Ticket_ID": uuid.uuid4().hex[:8],
                        "Device_ID": child,
                        "Timestamp": incident_time + timedelta(seconds=delay),
                        # Nếu là ONT thì báo Lost, Switch con thì báo Link Down
                        "Description": "ERROR_Lost 100%" if "ONT" in child else "Link Down",
                        "Is_Root_Cause": 0 
                    })
                    
        # 2. Sinh nhiễu (Noise - Tickets bình thường)
        daily_noise = np.random.poisson(CONFIG["NOISE_PER_DAY"])
        for _ in range(daily_noise):
            noise_node = random.choice(all_node_ids)
            noise_time = current_time + timedelta(minutes=random.randint(0, 1400))
            tickets.append({
                "Ticket_ID": uuid.uuid4().hex[:8],
                "Device_ID": noise_node,
                "Timestamp": noise_time,
                "Description": random.choice(NOISE_TEMPLATES),
                "Is_Root_Cause": 0
            })
            
        current_time += timedelta(days=1)
        print(f"Generating data for day: {current_time.date()}...")

    return pd.DataFrame(tickets)

if __name__ == "__main__":
    print(">>> Building Topology...")
    G, devices = build_topology()
    
    print(">>> Simulating Traffic & Tickets...")
    df_tickets = simulate_traffic(G, devices)
    
    # Save files
    pd.DataFrame(devices).to_csv("dataset_nodes_info.csv", index=False)
    
    # Save Edges
    edges_list = [{"Source": u, "Target": v} for u, v in G.edges()]
    pd.DataFrame(edges_list).to_csv("dataset_topology_edges.csv", index=False)
    
    # Save Tickets
    df_tickets.sort_values("Timestamp").to_csv("dataset_tickets.csv", index=False)
    
    print(f"✅ DONE! Generated {len(df_tickets)} tickets over {CONFIG['DAYS']} days.")
    print("Files created: dataset_nodes_info.csv, dataset_topology_edges.csv, dataset_tickets.csv")