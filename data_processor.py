# data_processor.py
import pandas as pd
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data
import config
import pickle
import os

class TelcoGraphDataset:
    def __init__(self, mode='train'):
        self.node_mapping = {}
        self.reverse_mapping = {}
        self.tfidf = TfidfVectorizer(max_features=config.TEXT_EMBEDDING_DIM, stop_words='english')
        self.type_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # 1. Load Data
        self.nodes_df = pd.read_csv(config.NODE_FILE)
        self.edges_df = pd.read_csv(config.EDGE_FILE)
        
        # Nếu file ticket tồn tại thì load, không thì để trống (cho trường hợp infer sau này)
        if os.path.exists(config.TICKET_FILE):
            self.tickets_df = pd.read_csv(config.TICKET_FILE)
            self.tickets_df['Timestamp'] = pd.to_datetime(self.tickets_df['Timestamp'])
        else:
            self.tickets_df = pd.DataFrame()

        # 2. Xử lý Vectorizer (FIT TRƯỚC khi dùng)
        if mode == 'train':
            print("Fitting Vectorizers on Training Data...")
            # Fit text features
            self.tfidf.fit(self.tickets_df['Description'].fillna(""))
            # Fit node static features
            self.type_encoder.fit(self.nodes_df[['type', 'vendor']])
            
            # Save vectorizers
            with open(config.VECTORIZER_PATH, 'wb') as f:
                pickle.dump((self.tfidf, self.type_encoder), f)
        else:
            # Mode eval/infer: Load vectorizers đã train
            if os.path.exists(config.VECTORIZER_PATH):
                print("Loading Vectorizers...")
                with open(config.VECTORIZER_PATH, 'rb') as f:
                    self.tfidf, self.type_encoder = pickle.load(f)
            else:
                raise Exception(f"Vectorizer file {config.VECTORIZER_PATH} not found! Run training first.")

        # 3. Sau khi đã có encoder, mới chuẩn bị mapping và static features
        self._prepare_mappings_and_static_data()

    def _prepare_mappings_and_static_data(self):
        # Map Node ID <-> Integer Index
        for idx, row in self.nodes_df.iterrows():
            self.node_mapping[row['id']] = idx
            self.reverse_mapping[idx] = row['id']
            
        # Xây dựng Edge Index (Cấu trúc đồ thị tĩnh)
        src, dst = [], []
        for _, row in self.edges_df.iterrows():
            if row['Source'] in self.node_mapping and row['Target'] in self.node_mapping:
                u, v = self.node_mapping[row['Source']], self.node_mapping[row['Target']]
                # Đồ thị vô hướng (2 chiều) để tin lan truyền tốt hơn
                src.extend([u, v])
                dst.extend([v, u])
        self.edge_index = torch.tensor([src, dst], dtype=torch.long)
        
        # Tạo Static Features (Type, Vendor) cho tất cả các node
        # Lúc này self.type_encoder ĐÃ ĐƯỢC FIT rồi, nên gọi transform sẽ không lỗi
        self.static_x = self.type_encoder.transform(self.nodes_df[['type', 'vendor']])

    def create_time_windows(self, window_size_min=10):
        """Chia toàn bộ ticket thành các cửa sổ thời gian"""
        if self.tickets_df.empty:
            return []
            
        start_time = self.tickets_df['Timestamp'].min()
        end_time = self.tickets_df['Timestamp'].max()
        
        windows = []
        current = start_time
        while current < end_time:
            next_window = current + pd.Timedelta(minutes=window_size_min)
            # Lọc ticket trong khoảng này
            mask = (self.tickets_df['Timestamp'] >= current) & (self.tickets_df['Timestamp'] < next_window)
            batch_df = self.tickets_df[mask]
            
            # Chỉ lấy window nào CÓ ticket (để tiết kiệm thời gian train)
            if not batch_df.empty:
                windows.append(batch_df)
            current = next_window
            
        return windows

    def df_to_graph_data(self, batch_df):
        """Chuyển đổi DataFrame ticket của 1 cửa sổ thời gian thành PyG Data"""
        num_nodes = len(self.nodes_df)
        labels = np.zeros(num_nodes, dtype=float)
        
        # Dynamic Features (Text Embedding) khởi tạo bằng 0
        dynamic_x = np.zeros((num_nodes, config.TEXT_EMBEDDING_DIM))
        
        # Map tickets vào node tương ứng
        for _, row in batch_df.iterrows():
            if row['Device_ID'] in self.node_mapping:
                idx = self.node_mapping[row['Device_ID']]
                
                # Vector hóa Description
                # Dùng transform (không fit lại)
                vec = self.tfidf.transform([row['Description']]).toarray()[0] # type: ignore
                dynamic_x[idx] += vec 
                
                # Gán nhãn Root Cause
                if row.get('Is_Root_Cause', 0) == 1:
                    labels[idx] = 1.0

        # Kết hợp Static Features và Dynamic Features
        # Static (ví dụ 10 chiều) + Dynamic (16 chiều) -> Feature Vector 26 chiều
        final_x = np.hstack([self.static_x, dynamic_x]) # type: ignore
        
        return Data(
            x=torch.tensor(final_x, dtype=torch.float),
            edge_index=self.edge_index,
            y=torch.tensor(labels, dtype=torch.float)
        )