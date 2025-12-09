# RCA-GNN Refactored Code Structure

This document describes the refactored code structure for the RCA-GNN (Root Cause Analysis using Graph Neural Networks) project.

## 📁 Project Structure

```
rca-gnn-demo/
├── data/
│   ├── generators/                  # Data generation modules
│   │   ├── __init__.py
│   │   ├── device_generator.py      # Device generation logic
│   │   ├── topology_generator.py    # Network topology generation
│   │   ├── event_generator.py       # Event generation
│   │   ├── incident_generator.py    # Incident & label generation (BFS)
│   │   └── customer_service_generator.py  # Customer/service data
│   ├── generate_samples_refactored.py     # Main data generation script
│   ├── generate_samples_v2.py       # Original data generation (legacy)
│   └── samples/                     # Generated CSV files
│       ├── devices.csv
│       ├── edges.csv
│       ├── events.csv
│       ├── incidents.csv
│       ├── node_labels.csv
│       ├── customers.csv
│       └── services.csv
│
├── src/
│   ├── config.py                    # Configuration constants
│   ├── utils.py                     # Utility functions
│   ├── main_refactored.py          # Refactored main entry point
│   ├── main.py                      # Original main (legacy)
│   │
│   ├── data/                        # Data loading & preprocessing
│   │   ├── __init__.py
│   │   └── data_loader.py           # CSV loading, timestamp parsing
│   │
│   ├── features/                    # Feature engineering
│   │   ├── __init__.py
│   │   └── feature_engineering.py   # Static & dynamic features
│   │
│   ├── graph/                       # Graph construction
│   │   ├── __init__.py
│   │   └── graph_builder.py         # Topology graph, sample building
│   │
│   ├── models/                      # Neural network models
│   │   ├── __init__.py
│   │   └── gat_model.py             # GAT architecture
│   │
│   ├── train/                       # Training & evaluation
│   │   ├── __init__.py
│   │   └── trainer.py               # Training loop, metrics
│   │
│   └── inference/                   # Inference & prediction
│       ├── __init__.py
│       └── predictor.py             # Root cause prediction
│
├── test/
│   └── together_llm.py              # LLM integration test
│
├── output/                          # Model checkpoints & metadata
│   ├── gat_rca_state.pt
│   └── meta.json
│
├── LICENSE
└── README.md
```

## 🔧 Module Descriptions

### Configuration (`src/config.py`)
- All constants and hyperparameters in one place
- Model architecture settings
- Training parameters
- Data generation parameters
- File paths and names

### Data Loading (`src/data/`)
- **data_loader.py**: CSV file loading, timestamp parsing, device indexing
- Clean separation of data I/O from business logic

### Feature Engineering (`src/features/`)
- **feature_engineering.py**: 
  - Static features (vendor, layer) with one-hot encoding
  - Dynamic features (event aggregation, critical count)
  - Topological features (node degree)
  - Feature combination utilities

### Graph Building (`src/graph/`)
- **graph_builder.py**:
  - NetworkX topology graph construction
  - PyTorch Geometric edge index conversion
  - Per-incident sample generation
  - Integration of features and labels

### Models (`src/models/`)
- **gat_model.py**:
  - GAT (Graph Attention Network) architecture
  - Multi-head attention layers
  - 3-class classifier (normal, victim, root)
  - Configurable architecture

### Training (`src/train/`)
- **trainer.py**:
  - Training loop with early stopping
  - Evaluation metrics (Top-1, Top-3 accuracy)
  - Per-class accuracy tracking
  - Train/test split management

### Inference (`src/inference/`)
- **predictor.py**:
  - Root cause prediction for new events
  - Time window-based inference
  - Top-k candidate ranking
  - Probability scoring

### Utilities (`src/utils.py`)
- Model saving/loading
- Metadata management
- Device selection (CPU/CUDA)
- Model summary printing

### Data Generation (`data/generators/`)
- **device_generator.py**: Random device creation
- **topology_generator.py**: Hierarchical network topology
- **event_generator.py**: Random event generation
- **incident_generator.py**: Incident creation with BFS labeling
- **customer_service_generator.py**: Customer and service data

## 🚀 Usage

### 1. Generate Sample Data

```bash
cd data
python generate_samples_refactored.py --output_dir ./samples
```

This generates:
- `devices.csv` - Network devices
- `edges.csv` - Network topology
- `events.csv` - System events
- `incidents.csv` - Incident records
- `node_labels.csv` - Ground truth labels (0=normal, 1=victim, 2=root)
- `customers.csv` - Customer data
- `services.csv` - Service data

### 2. Train the Model

```bash
cd src
python main_refactored.py --mode train --data_dir ../data/samples --epochs 40
```

Options:
- `--data_dir`: Directory with CSV files
- `--out_dir`: Output directory for model (default: `./output`)
- `--epochs`: Number of training epochs (default: 40)
- `--lr`: Learning rate (default: 0.001)
- `--window_mins`: Event aggregation window in minutes (default: 5)

### 3. Run Inference

```bash
cd src
python main_refactored.py --mode infer --data_dir ../data/samples --topk 5
```

Options:
- `--topk`: Number of top candidates (default: 3)
- `--infer_time`: ISO timestamp for inference (default: now)

## 📊 Key Improvements

### 1. **Modularity**
   - Each module has a single responsibility
   - Easy to test and maintain individual components
   - Clear separation of concerns

### 2. **Reusability**
   - Functions can be imported and used independently
   - Shared utilities in `config.py` and `utils.py`
   - DRY principle applied throughout

### 3. **Maintainability**
   - Comprehensive docstrings for all functions
   - Type hints for better IDE support
   - Consistent code style and structure

### 4. **Extensibility**
   - Easy to add new feature types
   - Simple to experiment with different model architectures
   - Pluggable components (encoders, graph builders, etc.)

### 5. **Documentation**
   - Clear module organization
   - Self-documenting code with meaningful names
   - Detailed comments for complex logic

## 🔄 Migration Guide

### From Original to Refactored

**Original:**
```python
python main.py --mode train --data_dir ./data/samples
```

**Refactored:**
```python
python main_refactored.py --mode train --data_dir ../data/samples
```

**Original:**
```python
python generate_samples_v2.py
```

**Refactored:**
```python
python generate_samples_refactored.py --output_dir ./samples
```

## 🧪 Testing

The modular structure makes it easy to test individual components:

```python
# Test data loading
from src.data import load_csvs
devices, edges, events, incidents, labels = load_csvs("./data/samples")

# Test feature engineering
from src.features import fit_static_encoders
enc_vendor, enc_layer = fit_static_encoders(devices)

# Test model
from src.models import GAT_RCA
model = GAT_RCA(in_channels=20)
```

## 📝 Configuration

All configuration is centralized in `src/config.py`:

```python
# Model hyperparameters
DEFAULT_HIDDEN_CHANNELS = 32
DEFAULT_GAT_HEADS_1 = 4
DEFAULT_DROPOUT = 0.2

# Training parameters
DEFAULT_EPOCHS = 40
DEFAULT_LEARNING_RATE = 1e-3

# Data parameters
DEFAULT_WINDOW_MINUTES = 5
```

## 🎯 Benefits of Refactored Structure

1. **Easier Debugging**: Isolate issues to specific modules
2. **Better Testing**: Unit test individual components
3. **Team Collaboration**: Multiple developers can work on different modules
4. **Code Reuse**: Import and use functions in notebooks or other scripts
5. **Clear Dependencies**: Easy to see what depends on what
6. **Professional Structure**: Follows Python best practices

## 📚 Next Steps

1. Add unit tests for each module
2. Create Jupyter notebooks for analysis
3. Add more model architectures in `src/models/`
4. Implement additional feature types in `src/features/`
5. Add logging throughout the codebase
6. Create Docker container for deployment

## 🤝 Contributing

When adding new features:
1. Follow the existing module structure
2. Add docstrings to all functions
3. Update this README with new modules
4. Keep configuration in `config.py`
5. Add type hints where appropriate

---

**Original Files (Legacy):**
- `src/main.py` - Original monolithic implementation
- `data/generate_samples_v2.py` - Original data generation

**Refactored Files (Use These):**
- `src/main_refactored.py` - Clean modular implementation
- `data/generate_samples_refactored.py` - Modular data generation
