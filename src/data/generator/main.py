import networkx as nx
import pandas as pd
import numpy as np
import random
import time
from faker import Faker
from datetime import datetime, timedelta

# Cấu hình hạt giống để tái lập kết quả
random.seed(42)
np.random.seed(42)
fake = Faker()

# --- CẤU HÌNH DANH MỤC THIẾT BỊ ---
DEVICE_CATALOG = {
    "CORE_ROUTER": {
        "Cisco": ["ASR9000", "NCS5500"],
        "Juniper": ["MX960", "PTX1000"]
    },
    "AGG_SWITCH": { # Switch DE/DI
        "Huawei": ["S6730", "CloudEngine 6800"],
        "H3C": ["S5560", "S6850"],
        "Cisco": ["Nexus 9300", "Catalyst 9500"]
    },
    "OLT": {
        "Huawei": ["MA5800-X7", "MA5600T"],
        "ZTE": ["C300", "C320"],
        "GCOM": ["GC16", "GL5610"],
        "Dasan": ["V8240", "V5812"]
    },
    "ONT": { # Thiết bị khách hàng
        "Huawei": ["HG8245", "EG8141"],
        "ZTE": ["F670Y", "F600"],
        "Dasand": ["H660GM"]
    }
}

class NetworkGenerator:
    def __init__(self, num_nodes_target=3000):
        self.G = nx.DiGraph() # Directed Graph (Uplink/Downlink)
        self.target = num_nodes_target
        
    def generate_topology(self):
        print(f"--- Đang tạo Topology mạng với mục tiêu ~{self.target} nodes ---")
        
        # 1. Tạo Core Layer (Ít)
        core_nodes = []
        for i in range(2): 
            node_id = f"CORE_{i:02d}"
            vendor = random.choice(list(DEVICE_CATALOG["CORE_ROUTER"].keys()))
            model = random.choice(DEVICE_CATALOG["CORE_ROUTER"][vendor])
            self.add_node(node_id, "CORE", vendor, model)
            core_nodes.append(node_id)

        # 2. Tạo Aggregation Layer (Switch DE/DI)
        agg_nodes = []
        num_aggs = 20
        for i in range(num_aggs):
            node_id = f"AGG_SW_{i:03d}"
            vendor = random.choice(list(DEVICE_CATALOG["AGG_SWITCH"].keys()))
            model = random.choice(DEVICE_CATALOG["AGG_SWITCH"][vendor])
            self.add_node(node_id, "AGGREGATION", vendor, model)
            
            # Nối lên Core (Redundancy)
            uplink = random.choice(core_nodes)
            self.add_edge(node_id, uplink, "100GE", "Uplink")
            agg_nodes.append(node_id)

        # 3. Tạo Access Layer (OLT)
        olt_nodes = []
        num_olts = 80 
        for i in range(num_olts):
            node_id = f"OLT_{i:03d}"
            vendor = random.choice(list(DEVICE_CATALOG["OLT"].keys()))
            model = random.choice(DEVICE_CATALOG["OLT"][vendor])
            self.add_node(node_id, "OLT", vendor, model)
            
            # Nối lên Aggregation
            uplink = random.choice(agg_nodes)
            self.add_edge(node_id, uplink, "10GE", "Uplink")
            olt_nodes.append(node_id)

        # 4. Tạo Customer Layer (ONT) - Số lượng lớn
        # Tính toán số lượng còn lại để đạt target
        remaining_nodes = self.target - len(core_nodes) - len(agg_nodes) - len(olt_nodes)
        print(f"--- Đang sinh {remaining_nodes} ONTs khách hàng... ---")
        
        for i in range(remaining_nodes):
            node_id = f"ONT_CUST_{i:05d}"
            # OLT thường đồng bộ hãng với ONT (ví dụ OLT Huawei đi với ONT Huawei)
            uplink_olt = random.choice(olt_nodes)
            olt_vendor = self.G.nodes[uplink_olt]['vendor']
            
            # Xử lý logic vendor tương thích (đơn giản hóa)
            ont_vendor = olt_vendor if olt_vendor in DEVICE_CATALOG["ONT"] else random.choice(list(DEVICE_CATALOG["ONT"].keys()))
            ont_model = random.choice(DEVICE_CATALOG["ONT"].get(ont_vendor, ["Generic_ONT"]))
            
            self.add_node(node_id, "ONT", ont_vendor, ont_model)
            self.add_edge(node_id, uplink_olt, "GPON", "Uplink")
            
        print(f"--- Đã xong! Tổng số Nodes: {self.G.number_of_nodes()}, Edges: {self.G.number_of_edges()} ---")
        return self.G

    def add_node(self, node_id, role, vendor, model):
        # Thêm thuộc tính phong phú
        firmware = f"v{random.randint(1,5)}.{random.randint(0,9)}"
        location = fake.city()
        self.G.add_node(node_id, 
                        label=role, # Dùng cho visualization
                        type=role,
                        vendor=vendor,
                        model=model,
                        firmware=firmware,
                        location=location,
                        ip=fake.ipv4())

    def add_edge(self, src, dst, link_type, direction):
        # Trong Graph RCA, hướng cạnh rất quan trọng (Dependency).
        # Nếu A (Switch) cung cấp mạng cho B (OLT), thì B phụ thuộc A.
        # Edge: B -> A (Dependency edge) hoặc A -> B (Physical traffic flow).
        # Ở đây ta dùng Dependency: ONT -> OLT -> AGG -> CORE (Con trỏ đến cha)
        self.G.add_edge(src, dst, type=link_type, direction=direction)

class LogSimulator:
    def __init__(self, graph):
        self.G = graph
        self.logs = []
        self.metrics = []
        
    def simulate_metric_normal(self, node_id, timestamp):
        # Sinh metric CPU/Optical Power bình thường
        node_attrs = self.G.nodes[node_id]
        if node_attrs['type'] == 'OLT':
            return {
                "timestamp": timestamp,
                "node_id": node_id,
                "cpu_usage": np.random.normal(30, 5), # Mean 30%, Std 5%
                "temp": np.random.normal(45, 2),
                "status": "NORMAL"
            }
        return None

    def inject_incident(self, root_cause_node, start_time):
        """
        Giả lập một sự cố:
        1. Root Cause Node bị lỗi (VD: High CPU -> Down).
        2. Lan truyền: Tất cả các Node con (descendants) bị mất kết nối (Alarm LOS).
        """
        incident_id = f"INC-{int(start_time.timestamp())}-{root_cause_node}"
        
        # 1. Log cho Root Cause
        root_log = {
            "timestamp": start_time,
            "incident_id": incident_id,
            "node_id": root_cause_node,
            "alarm_code": "CPU_OVERLOAD" if random.random() > 0.5 else "POWER_LOSS",
            "severity": "CRITICAL",
            "is_root_cause": True,  # LABEL QUAN TRỌNG ĐỂ TRAIN
            "node_type": self.G.nodes[root_cause_node]['type']
        }
        self.logs.append(root_log)
        
        # 2. Tìm các node chịu ảnh hưởng (Impact Analysis)
        # Trong graph này, edge là con -> cha (Dependency). Nên tìm node bị ảnh hưởng là tìm những node trỏ đến root_cause_node.
        # Tuy nhiên networkx ancestors/descendants phụ thuộc chiều mũi tên. 
        # Ta quy ước: ONT -> OLT -> AGG. Nếu AGG chết, OLT và ONT chết. -> Cần tìm những thằng trỏ vào AGG (predecessors).
        
        impacted_nodes = list(nx.ancestors(self.G, root_cause_node)) # Tìm ngược chiều mũi tên
        
        # Tạo nhiễu thời gian: Không phải alarm nổ cùng lúc, mà có delay
        for node in impacted_nodes:
            delay = random.randint(1, 60) # Delay 1s đến 60s
            log_time = start_time + timedelta(seconds=delay)
            
            symptom_log = {
                "timestamp": log_time,
                "incident_id": incident_id,
                "node_id": node,
                "alarm_code": "LOS" if self.G.nodes[node]['type'] == 'ONT' else "UPLINK_DOWN",
                "severity": "MAJOR",
                "is_root_cause": False, # Đây là triệu chứng
                "node_type": self.G.nodes[node]['type']
            }
            self.logs.append(symptom_log)
            
        return len(impacted_nodes)

    def run_simulation(self, duration_days=1, incidents_count=10):
        print(f"--- Đang giả lập log trong {duration_days} ngày với {incidents_count} sự cố ---")
        base_time = datetime.now() - timedelta(days=duration_days)
        
        # Chọn ngẫu nhiên node để gây lỗi (ưu tiên OLT và Aggregation để thấy lan truyền nhiều)
        potential_roots = [n for n, d in self.G.nodes(data=True) if d['type'] in ['OLT', 'AGGREGATION']]
        
        for _ in range(incidents_count):
            rc_node = random.choice(potential_roots)
            offset = random.randint(0, duration_days * 86400)
            event_time = base_time + timedelta(seconds=offset)
            
            impact_count = self.inject_incident(rc_node, event_time)
            # print(f"  + Sự cố tại {rc_node} ({self.G.nodes[rc_node]['type']}) -> Ảnh hưởng {impact_count} nodes")

        return pd.DataFrame(self.logs)

# --- THỰC THI ---

# 1. Tạo Mạng
gen = NetworkGenerator(num_nodes_target=2500)
G = gen.generate_topology()

# 2. Xuất dữ liệu Topology (Nodes & Edges) ra CSV
nodes_data = []
for n, attr in G.nodes(data=True):
    attr['id'] = n
    nodes_data.append(attr)
df_nodes = pd.DataFrame(nodes_data)
df_edges = nx.to_pandas_edgelist(G)

# 3. Giả lập Logs
sim = LogSimulator(G)
# Giả lập 7 ngày, 50 sự cố ngẫu nhiên
df_logs = sim.run_simulation(duration_days=7, incidents_count=50)

# Sắp xếp log theo thời gian
df_logs = df_logs.sort_values(by="timestamp")

print("\n--- SAMPLE DATA NODES ---")
print(df_nodes.head(3))
print("\n--- SAMPLE DATA LOGS (LABELED) ---")
print(df_logs[['timestamp', 'node_id', 'alarm_code', 'is_root_cause']].head(10))

# Lưu file để dùng sau
df_nodes.to_csv("network_nodes.csv", index=False)
df_edges.to_csv("network_edges.csv", index=False)
df_logs.to_csv("network_logs.csv", index=False)