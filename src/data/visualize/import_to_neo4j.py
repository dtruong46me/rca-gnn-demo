import pandas as pd
from neo4j import GraphDatabase
import time

# --- CẤU HÌNH KẾT NỐI ---
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "password123")

class Neo4jLoader:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self):
        self.driver.close()

    def clear_database(self):
        """Xóa sạch dữ liệu cũ để import mới (Cẩn thận khi dùng trên Production)"""
        print("--- Cleaning Database... ---")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Database cleared.")

    def create_constraints(self):
        """Tạo index để tìm kiếm nhanh và tránh duplicate"""
        print("--- Creating Indexes... ---")
        with self.driver.session() as session:
            # SỬA ĐỔI: Thêm 'IF NOT EXISTS' để script chạy đi chạy lại không bị lỗi
            
            # Tạo index cho ID thiết bị
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Device) REQUIRE n.id IS UNIQUE")
            
            # Tạo index cho Incident (Log)
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Incident) REQUIRE i.incident_id IS UNIQUE")

    def import_nodes(self, csv_path):
        """Import Nodes từ CSV sử dụng Batch UNWIND để tối ưu tốc độ"""
        print(f"--- Importing Nodes from {csv_path}... ---")
        df = pd.read_csv(csv_path)
        
        # Chuyển DataFrame thành List of Dictionaries
        nodes_data = df.to_dict('records')
        
        # Query Cypher: Sử dụng UNWIND để insert hàng loạt (Bulk Insert)
        # Chúng ta gán 2 Label: :Device (chung) và Label động (VD: :OLT, :ONT)
        query = """
        UNWIND $batch AS row
        MERGE (n:Device {id: row.id})
        SET n += row  -- Gán tất cả thuộc tính từ CSV vào Node (Vendor, Model, Firmware...)
        
        -- Hack trick để gán Label động trong Cypher (neo4j graph core)
        WITH n, row
        CALL apoc.create.addLabels(n, [row.type]) YIELD node
        RETURN count(*)
        """
        
        # Lưu ý: Nếu không có APOC plugin, ta dùng cách thủ công hơn bên dưới
        # Cách Standard (Không cần APOC):
        query_standard = """
        UNWIND $batch AS row
        MERGE (n:Device {id: row.id})
        SET n.vendor = row.vendor, 
            n.model = row.model, 
            n.firmware = row.firmware,
            n.location = row.location,
            n.type = row.type
        """

        # Chia nhỏ batch (ví dụ 1000 node mỗi lần gửi)
        batch_size = 1000
        with self.driver.session() as session:
            for i in range(0, len(nodes_data), batch_size):
                batch = nodes_data[i:i+batch_size]
                # Ở đây tôi dùng query_standard để bạn không phải cài thêm APOC
                session.run(query_standard, batch=batch)
                
                # Sau khi tạo node, chạy thêm lệnh để gán Label cụ thể (OLT, ONT...)
                # Vì Cypher cơ bản không cho parameter hóa Label (vd: :$Label là sai)
                for item in batch:
                    label = item['type']
                    node_id = item['id']
                    session.run(f"MATCH (n:Device {{id: $id}}) SET n:{label}", id=node_id) # type: ignore
                    
        print(f"Imported {len(nodes_data)} nodes.")

    def import_edges(self, csv_path):
        """Import Edges (Topology)"""
        print(f"--- Importing Edges from {csv_path}... ---")
        df = pd.read_csv(csv_path)
        
        # Chuẩn hóa tên cột nếu cần (trong code python trước tạo ra source, target)
        edges_data = df.to_dict('records')
        
        query = """
        UNWIND $batch AS row
        MATCH (source:Device {id: row.source})
        MATCH (target:Device {id: row.target})
        MERGE (source)-[r:CONNECTED_TO]->(target)
        SET r.type = row.type,
            r.direction = row.direction
        """
        
        batch_size = 1000
        with self.driver.session() as session:
            for i in range(0, len(edges_data), batch_size):
                batch = edges_data[i:i+batch_size]
                session.run(query, batch=batch)
        print(f"Imported {len(edges_data)} edges.")

    def import_logs_as_events(self, log_csv_path):
        """
        Biến Log thành Node sự kiện và nối vào thiết bị.
        (Device)-[:HAS_EVENT]->(Log)
        """
        print(f"--- Importing Logs from {log_csv_path}... ---")
        try:
            df = pd.read_csv(log_csv_path)
            # Chuyển đổi timestamp sang string hoặc format chuẩn nếu cần, 
            # nhưng Neo4j lưu được số nên để nguyên cũng ổn.
        except FileNotFoundError:
            print("Log file not found, skipping.")
            return

        logs_data = df.to_dict('records')
        
        # --- ĐÃ SỬA: Xóa comment gây lỗi trong chuỗi query ---
        query = """
        UNWIND $batch AS row
        MATCH (d:Device {id: row.node_id})
        MERGE (e:Event {incident_unique: row.incident_id + '_' + row.node_id})
        SET e.timestamp = row.timestamp,
            e.alarm_code = row.alarm_code,
            e.severity = row.severity,
            e.is_root_cause = row.is_root_cause,
            e.incident_group = row.incident_id
        
        MERGE (d)-[:HAS_EVENT]->(e)
        """
        
        # Link các event cùng incident_id với nhau (Correlation)
        # (Optional: chỉ chạy khi cần visualize sự liên quan)
        query_correlate = """
        MATCH (e1:Event), (e2:Event)
        WHERE e1.incident_group = e2.incident_group AND e1 <> e2
        MERGE (e1)-[:CORRELATED_WITH]-(e2)
        """

        batch_size = 1000
        with self.driver.session() as session:
            for i in range(0, len(logs_data), batch_size):
                batch = logs_data[i:i+batch_size]
                session.run(query, batch=batch)
            
            # Nếu dữ liệu log lớn (>10k dòng), bước này có thể lâu, cân nhắc bỏ qua nếu chỉ test import
            # print("Linking correlated events...")
            # session.run(query_correlate) 
            
        print(f"Imported {len(logs_data)} logs.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    import os
    __root__ = os.getcwd()
    NODE_CSV = os.path.join(__root__, "network_nodes.csv")
    EDGE_CSV = os.path.join(__root__, "network_edges.csv")
    LOG_CSV = os.path.join(__root__, "network_logs.csv")


    # Đảm bảo bạn đã chạy file tạo dữ liệu trước đó để có các file .csv
    loader = Neo4jLoader(URI, AUTH)
    
    loader.clear_database()
    loader.create_constraints()
    
    # 1. Nạp Nodes (Thiết bị)
    loader.import_nodes(NODE_CSV)
    
    # 2. Nạp Edges (Dây cáp/Kết nối)
    loader.import_edges(EDGE_CSV)
    
    # 3. Nạp Logs (Nếu bạn muốn visualize cả log trên graph)
    loader.import_logs_as_events(LOG_CSV)
    
    print("\n=== SUCCESS! ===")
    print("Truy cập http://localhost:7474 để xem kết quả.")
    print("Query gợi ý: MATCH (n) RETURN n LIMIT 100")
    