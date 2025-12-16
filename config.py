# config.py
import torch

# Cấu hình tham số Model
HIDDEN_CHANNELS = 64
LEARNING_RATE = 0.001
EPOCHS = 50
DROPOUT = 0.2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# File paths
NODE_FILE = 'dataset_nodes_info.csv'
EDGE_FILE = 'dataset_topology_edges.csv'
TICKET_FILE = 'dataset_tickets.csv'
MODEL_PATH = 'rca_gnn_model.pth'
VECTORIZER_PATH = 'vectorizer.pkl'

# Cấu hình Feature
# Kích thước vector cho text log (Description)
TEXT_EMBEDDING_DIM = 16