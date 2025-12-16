import networkx as nx
import pandas as pd
import numpy as np
import random
import time
from faker import Faker
from datetime import datetime, timedelta
from pyvis.network import Network
import matplotlib.pyplot as plt

# --- CẤU HÌNH & KHỞI TẠO ---
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

# --- CLASS TẠO MẠNG (Phần 1 cũ) ---
class NetworkGenerator:
    def __init__(self, num_nodes_target=1000): # Giảm xuống 1000 cho nhẹ máy khi vẽ
        self.G = nx.DiGraph()
        self.target = num_nodes_target
        
    def generate_topology(self):
        print(f"--- 1. Đang tạo Topology mạng với mục tiêu ~{self.target} nodes ---")
        
        # 1. Core
        core_nodes = []
        for i in range(2): 
            node_id = f"CORE_{i:02d}"
            vendor = random.choice(list(DEVICE_CATALOG["CORE_ROUTER"].keys()))
            self.add_node(node_id, "CORE", vendor, random.choice(DEVICE_CATALOG["CORE_ROUTER"][vendor]))
            core_nodes.append(node_id)

        # 2. Aggregation
        agg_nodes = []
        for i in range(10): # 10 Agg switches
            node_id = f"AGG_SW_{i:03d}"
            vendor = random.choice(list(DEVICE_CATALOG["AGG_SWITCH"].keys()))
            self.add_node(node_id, "AGGREGATION", vendor, random.choice(DEVICE_CATALOG["AGG_SWITCH"][vendor]))
            self.add_edge(node_id, random.choice(core_nodes), "100GE", "Uplink")
            agg_nodes.append(node_id)

        # 3. OLT
        olt_nodes = []
        for i in range(30): # 30 OLTs
            node_id = f"OLT_{i:03d}"
            vendor = random.choice(list(DEVICE_CATALOG["OLT"].keys()))
            self.add_node(node_id, "OLT", vendor, random.choice(DEVICE_CATALOG["OLT"][vendor]))
            self.add_edge(node_id, random.choice(agg_nodes), "10GE", "Uplink")
            olt_nodes.append(node_id)

        # 4. Customer ONT
        remaining_nodes = self.target - len(core_nodes) - len(agg_nodes) - len(olt_nodes)
        print(f"--- Đang sinh {remaining_nodes} ONTs khách hàng... ---")
        
        for i in range(remaining_nodes):
            node_id = f"ONT_CUST_{i:05d}"
            uplink_olt = random.choice(olt_nodes)
            olt_vendor = self.G.nodes[uplink_olt]['vendor']
            ont_vendor = olt_vendor if olt_vendor in DEVICE_CATALOG["ONT"] else random.choice(list(DEVICE_CATALOG["ONT"].keys()))
            
            self.add_node(node_id, "ONT", ont_vendor, "Generic_ONT")
            self.add_edge(node_id, uplink_olt, "GPON", "Uplink")
            
        print(f"--- Đã xong! Total Nodes: {self.G.number_of_nodes()} ---")
        return self.G

    def add_node(self, node_id, role, vendor, model):
        self.G.add_node(node_id, label=role, type=role, vendor=vendor, model=model, 
                        firmware=f"v{random.randint(1,5)}", location=fake.city())

    def add_edge(self, src, dst, link_type, direction):
        self.G.add_edge(src, dst, type=link_type, direction=direction)

# --- HÀM VISUALIZATION (Phần bạn vừa paste) ---
def visualize_topology_interactive(G, filename="network_map.html"):
    print(f"--- 2. Đang vẽ Topology 2D Interactive vào file {filename} ---")
    
    # --- THAY ĐỔI 1: Full màn hình ---
    # height="100vh": Chiếm 100% chiều cao cửa sổ trình duyệt
    # width="100%": Chiếm 100% chiều rộng
    net = Network(height="95vh", width="100%", bgcolor="#222222", font_color="white", select_menu=True, cdn_resources='remote') # type: ignore
    
    color_map = { "CORE": "#ff4d4d", "AGGREGATION": "#ffa600", "OLT": "#00b4d8", "ONT": "#90be6d" }

    # --- THAY ĐỔI 2: Xem Full Graph (Bỏ Sampling) ---
    # Nếu máy yếu hoặc trình duyệt bị đơ, hãy uncomment đoạn code sampling cũ.
    # Ở đây tôi dùng trực tiếp G (toàn bộ nodes)
    H = G 

    print(f"   + Đang render tổng cộng {H.number_of_nodes()} nodes và {H.number_of_edges()} edges...")

    for node, data in H.nodes(data=True):
        color = color_map.get(data.get('type', 'ONT'), "#ffffff")
        
        # Tạo Tooltip chi tiết
        title_html = (
            f"<b>ID:</b> {node}<br>"
            f"<b>Type:</b> {data.get('type', 'N/A')}<br>"
            f"<b>Vendor:</b> {data.get('vendor', 'N/A')}<br>"
            f"<b>Model:</b> {data.get('model', 'N/A')}<br>"
            f"<b>Loc:</b> {data.get('location', 'N/A')}"
        )
        
        # Size node
        size = 60 if data.get('type') == 'CORE' else (30 if data.get('type') == 'AGGREGATION' else (20 if data.get('type') == 'OLT' else 10))
        
        net.add_node(node, label=node, title=title_html, color=color, size=size)

    for src, dst, data in H.edges(data=True):
        # Mũi tên chỉ hướng đi của tín hiệu/quan hệ
        net.add_edge(src, dst, color="#555555", width=1, arrowStrikethrough=False)

    # Cấu hình Physics: Chọn 'barnes_hut' để render nhanh hơn cho graph lớn
    # force_atlas_2based đẹp nhưng rất nặng với >1000 nodes
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=95, spring_strength=0.001, damping=0.09, overlap=0)
    
    # --- THAY ĐỔI 3: Bỏ thanh điều khiển (Buttons) ---
    # Đã xóa dòng net.show_buttons(...)
    
    net.save_graph(filename)
    print(f"Done! Mở file {filename} để xem.")

# --- MAIN EXECUTION BLOCK (Chạy chương trình) ---
if __name__ == "__main__":
    # BƯỚC 1: TẠO DỮ LIỆU (Generate G)
    # Đây là bước bạn bị thiếu -> G chưa được định nghĩa
    gen = NetworkGenerator(num_nodes_target=1000)
    G = gen.generate_topology() 

    # BƯỚC 2: VẼ (Visualize G)
    # Lúc này G đã có dữ liệu, truyền vào hàm mới chạy được
    visualize_topology_interactive(G, filename="network_map.html")