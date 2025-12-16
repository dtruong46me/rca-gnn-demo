import pandas as pd
import networkx as nx
import random
import uuid
from datetime import datetime, timedelta
import numpy as np

# ==========================================
# 1. CẤU HÌNH (CONFIGURATION)
# ==========================================
CONFIG = {
    "NUM_CORE_SWITCHES": 2,       # Số lượng Core (Juniper, Cisco)
    "NUM_AGG_SWITCHES": 10,       # Số lượng Aggregation (Cisco, H3C)
    "NUM_ACCESS_SWITCHES": 50,    # Số lượng Access/OLT (Huawei, ZTE, GCOM)
    "NUM_ONTS": 2000,             # Số lượng Modem nhà khách hàng
    "START_TIME": datetime(2025, 12, 8, 8, 0, 0),
    "DURATION_MINUTES": 240,      # Giả lập trong 4 tiếng
    
    # Cấu hình sự cố
    "NUM_INCIDENTS": 3,           # Số lượng sự cố Root Cause giả lập
    "PROPAGATION_RATE": 0.8,      # 80% thiết bị con sẽ báo lỗi khi cha chết
    "NOISE_TICKETS": 500,         # Số lượng ticket rác (bảo trì, cấu hình...)
}

# Danh sách Vendor và Model
VENDORS = {
    "CORE": ["Juniper MX960", "Cisco ASR9000"],
    "AGG": ["Cisco Nexus", "H3C S12500", "Huawei S9300"],
    "ACCESS": ["Huawei MA5600T", "ZTE C320", "GCOM", "Alcatel-Lucent"],
    "ONT": ["Huawei HG8145", "ZTE F670", "Dasan"]
}

# Các mẫu mô tả lỗi (Templates) lấy từ yêu cầu của bạn
ERROR_TEMPLATES = [
    "ERROR_Lost 100%", 
    "cảnh báo CRITICAL - {ip}: rta nan, lost 100%",
    "Interface Down", 
    "High Optical Loss (-35dBm)",
    "BGP Neighbor Down",
    "OSPF State Change to Down"
]

NORMAL_TEMPLATES = [
    "Bảo trì có kế hoạch",
    "Cấu hình PON {device_name}",
    "Nhờ check port module quang",
    "Khách hàng báo mạng chậm",
    "Thay đổi cấu hình VLAN"
]

# ==========================================
# 2. XÂY DỰNG TOPOLOGY (GRAPH)
# ==========================================
def build_network_topology():
    G = nx.DiGraph() # Đồ thị có hướng (Cha -> Con)
    devices = []
    
    # 1. Tạo Core Layer
    cores = []
    for i in range(CONFIG["NUM_CORE_SWITCHES"]):
        dev_id = f"HN-CORE-{i+1:02d}"
        vendor = random.choice(VENDORS["CORE"])
        info = {"id": dev_id, "type": "CORE", "vendor": vendor, "ip": f"10.0.0.{i+1}"}
        devices.append(info)
        cores.append(dev_id)
        G.add_node(dev_id, **info)

    # 2. Tạo Aggregation Layer
    aggs = []
    for i in range(CONFIG["NUM_AGG_SWITCHES"]):
        dev_id = f"HN-AGG-{i+1:03d}"
        vendor = random.choice(VENDORS["AGG"])
        info = {"id": dev_id, "type": "AGG", "vendor": vendor, "ip": f"10.1.{i//255}.{i%255}"}
        devices.append(info)
        aggs.append(dev_id)
        G.add_node(dev_id, **info)
        # Nối vào Core ngẫu nhiên
        parent = random.choice(cores)
        G.add_edge(parent, dev_id)

    # 3. Tạo Access Layer (OLT/Switch)
    access_devs = []
    for i in range(CONFIG["NUM_ACCESS_SWITCHES"]):
        dev_id = f"HN-OLT-{i+1:04d}" # Đặt tên kiểu OLT
        vendor = random.choice(VENDORS["ACCESS"])
        info = {"id": dev_id, "type": "ACCESS", "vendor": vendor, "ip": f"172.16.{i//255}.{i%255}"}
        devices.append(info)
        access_devs.append(dev_id)
        G.add_node(dev_id, **info)
        # Nối vào Agg ngẫu nhiên
        parent = random.choice(aggs)
        G.add_edge(parent, dev_id)

    # 4. Tạo ONT Layer (Khách hàng)
    for i in range(CONFIG["NUM_ONTS"]):
        dev_id = f"ONT-KH-{uuid.uuid4().hex[:8].upper()}"
        vendor = random.choice(VENDORS["ONT"])
        info = {"id": dev_id, "type": "ONT", "vendor": vendor, "ip": "dynamic"}
        devices.append(info)
        G.add_node(dev_id, **info)
        # Nối vào Access/OLT ngẫu nhiên
        parent = random.choice(access_devs)
        G.add_edge(parent, dev_id)

    print(f"✅ Đã tạo Topology: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G, devices

# ==========================================
# 3. GIẢ LẬP SỰ CỐ & TICKET (SIMULATION)
# ==========================================
def generate_tickets(G, devices_list):
    tickets = []
    current_time = CONFIG["START_TIME"]
    
    # --- PHẦN 1: TẠO SỰ CỐ GỐC & BÃO CẢNH BÁO (ALARM STORM) ---
    root_causes = []
    
    # Chọn ngẫu nhiên thiết bị Aggregation hoặc Access làm Root Cause
    potential_roots = [d for d in devices_list if d["type"] in ["AGG", "ACCESS"]]
    
    for _ in range(CONFIG["NUM_INCIDENTS"]):
        # 1. Chọn Root Cause
        root_node = random.choice(potential_roots)
        root_causes.append(root_node["id"])
        
        # Thời điểm xảy ra lỗi
        incident_time = current_time + timedelta(minutes=random.randint(10, CONFIG["DURATION_MINUTES"]-60))
        
        # Tạo Ticket cho Root Cause (Label = 1)
        root_ticket = create_ticket_entry(root_node, incident_time, is_root=True)
        tickets.append(root_ticket)
        
        # 2. Lan truyền (Propagation) - Tìm tất cả con cháu
        # Sử dụng DFS để tìm tất cả các node bị ảnh hưởng downstream
        try:
            descendants = list(nx.descendants(G, root_node["id"]))
        except:
            descendants = []
            
        print(f"🔥 Incident tại {root_node['id']} ({root_node['type']}) -> Ảnh hưởng {len(descendants)} thiết bị con.")

        # Tạo ticket cho các thiết bị con (Symptom - Label = 0)
        for child_id in descendants:
            # Không phải con nào cũng báo lỗi (theo tỷ lệ propagation)
            if random.random() < CONFIG["PROPAGATION_RATE"]:
                child_node = G.nodes[child_id]
                # Thời gian trễ ngẫu nhiên (1-5 phút sau Root Cause)
                delay = random.randint(1, 300) 
                symptom_time = incident_time + timedelta(seconds=delay)
                
                symptom_ticket = create_ticket_entry(child_node, symptom_time, is_root=False, cause_node=root_node["id"])
                tickets.append(symptom_ticket)

    # --- PHẦN 2: TẠO TICKETS NHIỄU (NOISE/NORMAL) ---
    for _ in range(CONFIG["NOISE_TICKETS"]):
        rand_node = random.choice(devices_list)
        # Random thời gian
        rand_time = current_time + timedelta(minutes=random.randint(0, CONFIG["DURATION_MINUTES"]))
        
        # Tạo ticket loại Normal/Maintenance
        noise_ticket = create_ticket_entry_normal(rand_node, rand_time)
        tickets.append(noise_ticket)
        
    return tickets, root_causes

def create_ticket_entry(node_info, timestamp, is_root=False, cause_node=None):
    """Tạo ticket dạng lỗi"""
    ticket_id = f"SC{timestamp.strftime('%d%m%y')}{random.randint(10000, 99999)}"
    
    desc_template = random.choice(ERROR_TEMPLATES)
    description = desc_template.replace("{ip}", node_info.get("ip", "0.0.0.0")).replace("{device}", node_info["id"])
    
    return {
        "Ticket_ID": ticket_id,
        "Device_ID": node_info["id"],
        "Device_Type": node_info["type"],
        "Vendor": node_info["vendor"],
        "Timestamp": timestamp.isoformat(),
        "Description": description,
        "Status": "Closed", # Giả lập là đã đóng sau khi xử lý
        "Cause_Category": "Hardware Failure" if is_root else "Transmission/Power", # Root thì là Hardware, Con thì là đường truyền
        "Is_Root_Cause": 1 if is_root else 0, # LABEL QUAN TRỌNG CHO GNN
        "Linked_Root_Node": node_info["id"] if is_root else cause_node # Để kiểm tra debug
    }

def create_ticket_entry_normal(node_info, timestamp):
    """Tạo ticket bình thường/nhiễu"""
    ticket_id = f"HT{timestamp.strftime('%d%m%y')}{random.randint(10000, 99999)}"
    
    desc_template = random.choice(NORMAL_TEMPLATES)
    description = desc_template.replace("{device_name}", node_info["id"])
    
    return {
        "Ticket_ID": ticket_id,
        "Device_ID": node_info["id"],
        "Device_Type": node_info["type"],
        "Vendor": node_info["vendor"],
        "Timestamp": timestamp.isoformat(),
        "Description": description,
        "Status": "Closed",
        "Cause_Category": "Planned Maintenance" if "Bảo trì" in description else "Configuration",
        "Is_Root_Cause": 0, # Luôn là 0
        "Linked_Root_Node": None
    }

# ==========================================
# 4. MAIN & EXPORT
# ==========================================
if __name__ == "__main__":
    # 1. Build Graph
    G, devices_list = build_network_topology()
    
    # 2. Simulate Events
    tickets_data, roots = generate_tickets(G, devices_list)
    
    # 3. Convert to DataFrames
    df_tickets = pd.DataFrame(tickets_data)
    
    # Tạo danh sách Edges (Source -> Target)
    edges = list(G.edges())
    df_edges = pd.DataFrame(edges, columns=["Source_Device", "Target_Device"])
    
    # Tạo danh sách Nodes (Features)
    df_nodes = pd.DataFrame(devices_list)

    # 4. Save to CSV
    print(f"\n📊 Tổng hợp dữ liệu:")
    print(f"- Tổng số Tickets: {len(df_tickets)}")
    print(f"- Số lượng Root Cause Tickets (Lỗi gốc): {df_tickets['Is_Root_Cause'].sum()}")
    print(f"- Số lượng Symptom Tickets (Lỗi ăn theo): {len(df_tickets[(df_tickets['Is_Root_Cause']==0) & (df_tickets['Cause_Category']!='Planned Maintenance') & (df_tickets['Cause_Category']!='Configuration')])}")
    
    df_tickets.sort_values(by="Timestamp").to_csv("dataset_tickets.csv", index=False)
    df_edges.to_csv("dataset_topology_edges.csv", index=False)
    df_nodes.to_csv("dataset_nodes_info.csv", index=False)
    
    print("\n✅ Đã lưu 3 file: dataset_tickets.csv, dataset_topology_edges.csv, dataset_nodes_info.csv")
    print("👉 Hãy dùng 3 file này để xây dựng GraphDataset cho GNN.")