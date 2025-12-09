"""
Configuration settings for RCA-GNN system.
Contains all constants, hyperparameters, and paths used across the project.
"""

# Model hyperparameters
DEFAULT_HIDDEN_CHANNELS = 32
DEFAULT_GAT_HEADS_1 = 4
DEFAULT_GAT_HEADS_2 = 2
DEFAULT_OUTPUT_CLASSES = 3  # 0: normal, 1: victim, 2: root
DEFAULT_DROPOUT = 0.2

# Training parameters
DEFAULT_EPOCHS = 40
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_TEST_SIZE = 0.2
RANDOM_SEED = 42

# Data processing parameters
DEFAULT_WINDOW_MINUTES = 5
DEFAULT_TOP_K = 3

# File paths
DEFAULT_DATA_DIR = "."
DEFAULT_OUTPUT_DIR = "./output"
MODEL_STATE_FILE = "gat_rca_state.pt"
META_FILE = "meta.json"

# CSV file names
CSV_DEVICES = "devices.csv"
CSV_EDGES = "edges.csv"
CSV_EVENTS = "events.csv"
CSV_INCIDENTS = "incidents.csv"
CSV_LABELS = "node_labels.csv"
CSV_CUSTOMERS = "customers.csv"
CSV_SERVICES = "services.csv"

# Label definitions
LABEL_NORMAL = 0
LABEL_VICTIM = 1
LABEL_ROOT = 2

# Data generation parameters (for generate_samples)
NUM_DEVICES = 80
NUM_CUSTOMERS = 150
NUM_SERVICES = 80
NUM_EVENTS = 1500
NUM_INCIDENTS = 30

# Device attributes
VENDORS = ["Huawei", "ZTE", "GCOM", "Cisco", "Juniper"]
LAYERS = ["CORE", "AGG", "ACCESS", "OLT", "ONU"]
MODELS = ["X6000", "C300", "MA5800", "S6720", "QFX5100"]
EVENT_TYPES = ["LOS", "linkDown", "highCPU", "highTemp", "packetDrop"]
SEVERITY_LIST = ["critical", "major", "minor", "warning"]
LINK_TYPES = ["fiber", "ethernet"]

# Topology parameters
MAX_EDGES_PER_NODE = 3
LINK_CAPACITIES_MBPS = [100, 1000, 10000]

# Service types
SERVICE_TYPES = ["Internet", "VoIP", "VPN"]
