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
    "NUM_ONT": 3000,          # 3000 khách hàng
    "START_TIME": datetime(2025, 12, 1, 0, 0, 0),
    "DAYS": 30,               # Giả lập 30 ngày
    "INCIDENTS_PER_DAY": 5,   # Mỗi ngày có 5 sự cố mạng nghiêm trọng
    "NOISE_PER_DAY": 200,     # Mỗi ngày có 200 ticket rác/bảo trì
    "PROPAGATION_RATE": 0.7,  # 70% thiết bị con sẽ báo lỗi khi cha chết
}

VENDORS = {
    "CORE": ["Juniper MX960", "Cisco ASR9000"],
    "AGG": ["Cisco Nexus", "H3C S12500", "Huawei S9300"],
    "ACCESS": ["Huawei MA5600T", "ZTE C320", "GCOM"],
    "ONT": ["Huawei HG8145", "ZTE F670", "Dasan"]
}

ERROR_TEMPLATES = [
    "OSPF State Change to Down", "BGP Neighbor Down", "Interface Down",
    "High Optical Loss (-35dBm)", "ERROR_Lost 100%", "CPU Overload 99%"
]
NOISE_TEMPLATES = [
    "Bảo trì có kế hoạch", "Cấu hình VLAN", "Kiểm tra tín hiệu", 
    "Khách hàng báo chậm", "Thay đổi gói cước", "Reset port theo yêu cầu"
]

def build_topology():
    G = nx.DiGraph()
    devices = []
    
    # Tạo Core -> Agg -> Access -> ONT
    # (Code giữ nguyên logic cũ nhưng scale to hơn)
    cores = [f"CORE-{i}" for i in range(CONFIG["NUM_CORE"])]
    for d in cores: devices.append({"id": d, "type": "CORE", "vendor": random.choice(VENDORS["CORE"])})
    
    aggs = [f"AGG-{i}" for i in range(CONFIG["NUM_AGG"])]
    for d in aggs: 
        devices.append({"id": d, "type": "AGG", "vendor": random.choice(VENDORS["AGG"])})
        G.add_edge(random.choice(cores), d) # Nối lên Core
        
    accesses = [f"ACCESS-{i}" for i in range(CONFIG["NUM_ACCESS"])]
    for d in accesses:
        devices.append({"id": d, "type": "ACCESS", "vendor": random.choice(VENDORS["ACCESS"])})
        G.add_edge(random.choice(aggs), d) # Nối lên Agg
        
    onts = [f"ONT-{i}" for i in range(CONFIG["NUM_ONT"])]
    for d in onts:
        devices.append({"id": d, "type": "ONT", "vendor": random.choice(VENDORS["ONT"])})
        G.add_edge(random.choice(accesses), d) # Nối lên Access

    return G, devices

def simulate_traffic(G, devices):
    tickets = []
    current_time = CONFIG["START_TIME"]
    end_time = current_time + timedelta(days=CONFIG["DAYS"])
    
    root_nodes = [d["id"] for d in devices if d["type"] in ["AGG", "ACCESS"]] # Chỉ Agg/Access hay hỏng
    
    while current_time < end_time:
        # 1. Sinh sự cố (Incidents)
        daily_incidents = np.random.poisson(CONFIG["INCIDENTS_PER_DAY"])
        for _ in range(daily_incidents):
            # Chọn root cause
            root_id = random.choice(root_nodes)
            incident_time = current_time + timedelta(minutes=random.randint(0, 1400))
            
            # Ticket cho Root Cause
            tickets.append({
                "Ticket_ID": uuid.uuid4().hex[:8],
                "Device_ID": root_id,
                "Timestamp": incident_time,
                "Description": random.choice(ERROR_TEMPLATES),
                "Is_Root_Cause": 1
            })
            
            # Lan truyền lỗi xuống con cháu
            descendants = list(nx.descendants(G, root_id))
            for child in descendants:
                if random.random() < CONFIG["PROPAGATION_RATE"]:
                    delay = random.randint(1, 300) # Trễ 1-5 phút
                    tickets.append({
                        "Ticket_ID": uuid.uuid4().hex[:8],
                        "Device_ID": child,
                        "Timestamp": incident_time + timedelta(seconds=delay),
                        "Description": "ERROR_Lost 100%" if "ONT" in child else "Link Down",
                        "Is_Root_Cause": 0 # Nạn nhân
                    })
                    
        # 2. Sinh nhiễu (Noise)
        daily_noise = np.random.poisson(CONFIG["NOISE_PER_DAY"])
        for _ in range(daily_noise):
            noise_node = random.choice(devices)["id"]
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
    G, devices = build_topology()
    df_tickets = simulate_traffic(G, devices)
    
    # Save
    pd.DataFrame(devices).to_csv("dataset_nodes_info.csv", index=False)
    pd.DataFrame(list(G.edges()), columns=["Source", "Target"]).to_csv("dataset_topology_edges.csv", index=False)
    df_tickets.sort_values("Timestamp").to_csv("dataset_tickets.csv", index=False)
    print(f"✅ DONE! Generated {len(df_tickets)} tickets over {CONFIG['DAYS']} days.")