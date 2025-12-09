# Quick Reference Guide - RCA-GNN Refactored

## 📂 File Locations

### Main Scripts
- **Training/Inference**: `src/main_refactored.py`
- **Data Generation**: `data/generate_samples_refactored.py`
- **Configuration**: `src/config.py`

### Core Modules
```
src/
├── config.py              # All settings & constants
├── utils.py               # Helper functions
├── data/                  # Data loading
├── features/              # Feature engineering
├── graph/                 # Graph construction
├── models/                # Neural network models
├── train/                 # Training logic
└── inference/             # Prediction logic
```

## 🚀 Common Commands

### Generate Sample Data
```bash
cd data
python generate_samples_refactored.py --output_dir ./samples
```

### Train Model
```bash
cd src
python main_refactored.py --mode train --data_dir ../data/samples --epochs 40
```

### Run Inference
```bash
cd src
python main_refactored.py --mode infer --data_dir ../data/samples --topk 5
```

## 📝 Import Examples

### Load Data
```python
from src.data import load_csvs, build_device_index

devices, edges, events, incidents, labels = load_csvs("./data/samples")
device_list, device_index = build_device_index(devices)
```

### Feature Engineering
```python
from src.features import (
    fit_static_encoders,
    build_static_feature_matrix,
    aggregate_events_for_window
)

# Fit encoders
enc_vendor, enc_layer = fit_static_encoders(devices)

# Build features
vendor_mat, layer_mat = build_static_feature_matrix(
    devices, device_list, device_index, enc_vendor, enc_layer
)
```

### Build Graph
```python
from src.graph import build_samples

samples, edge_index, incident_ids = build_samples(
    devices, edges, events, incidents, labels,
    device_list, device_index,
    window_mins=5
)
```

### Create Model
```python
from src.models import GAT_RCA

in_dim = samples[0].x.shape[1]
model = GAT_RCA(in_channels=in_dim)
```

### Train
```python
from src.train import train_model

model, info = train_model(
    samples,
    in_dim,
    epochs=40,
    lr=0.001,
    device='cuda'
)
```

### Inference
```python
from src.inference import infer_from_events

results, probs = infer_from_events(
    model, devices, edges, events,
    device_list, device_index, edge_index,
    enc_vendor, enc_layer,
    window_center_time,
    topk=5
)
```

## 🔧 Configuration

### Model Parameters (in `config.py`)
```python
DEFAULT_HIDDEN_CHANNELS = 32
DEFAULT_GAT_HEADS_1 = 4
DEFAULT_GAT_HEADS_2 = 2
DEFAULT_DROPOUT = 0.2
```

### Training Parameters
```python
DEFAULT_EPOCHS = 40
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_TEST_SIZE = 0.2
```

### Data Generation
```python
NUM_DEVICES = 80
NUM_INCIDENTS = 30
NUM_EVENTS = 1500
```

## 📊 Output Files

### After Training
```
output/
├── gat_rca_state.pt    # Model weights
└── meta.json           # Device mapping metadata
```

### After Data Generation
```
data/samples/
├── devices.csv         # Network devices
├── edges.csv           # Network topology
├── events.csv          # System events
├── incidents.csv       # Incident records
├── node_labels.csv     # Ground truth labels
├── customers.csv       # Customer data
└── services.csv        # Service data
```

## 🎯 Key Functions by Task

### Task: Load and Prepare Data
1. `load_csvs()` - Load CSV files
2. `preprocess_timestamps()` - Parse dates
3. `build_device_index()` - Create index mapping

### Task: Engineer Features
1. `fit_static_encoders()` - Fit OneHot encoders
2. `build_static_feature_matrix()` - Encode vendor/layer
3. `aggregate_events_for_window()` - Count events
4. `compute_degree_features()` - Node degrees
5. `build_combined_features()` - Concatenate all

### Task: Build Graph
1. `build_topology_graph()` - Create NetworkX graph
2. `build_edge_index()` - Convert to PyG format
3. `build_samples()` - Create training samples

### Task: Train Model
1. `GAT_RCA()` - Initialize model
2. `train_model()` - Train with validation
3. `evaluate_model()` - Compute metrics
4. `save_model_and_metadata()` - Save results

### Task: Run Inference
1. `load_model_state()` - Load trained model
2. `infer_from_events()` - Predict root causes
3. `predict_root_causes()` - Simple prediction

## 📋 Module Checklist

When adding new features:

- [ ] Update `config.py` with new constants
- [ ] Add function to appropriate module
- [ ] Add docstring with type hints
- [ ] Update `__init__.py` in module
- [ ] Test function independently
- [ ] Update documentation

## 🐛 Debugging Tips

### Check Data Loading
```python
from src.data import load_csvs
devices, edges, events, incidents, labels = load_csvs("./data/samples")
print(f"Devices: {len(devices)}")
print(f"Edges: {len(edges)}")
print(f"Events: {len(events)}")
```

### Check Feature Dimensions
```python
print(f"Vendor features: {vendor_mat.shape}")
print(f"Layer features: {layer_mat.shape}")
print(f"Combined features: {X.shape}")
```

### Check Sample Structure
```python
sample = samples[0]
print(f"Node features: {sample.x.shape}")
print(f"Edge index: {sample.edge_index.shape}")
print(f"Labels: {sample.y.shape}")
```

### Check Model Output
```python
with torch.no_grad():
    logits = model(sample.x, sample.edge_index)
    print(f"Output shape: {logits.shape}")  # Should be [num_nodes, 3]
```

## 💡 Best Practices

1. **Always use config.py** for constants
2. **Import from package level** when possible
3. **Use type hints** in function signatures
4. **Add docstrings** to all functions
5. **Keep functions small** and focused
6. **Test modules independently** before integration

## 🔗 Related Files

- `REFACTORING_GUIDE.md` - Full refactoring documentation
- `ARCHITECTURE.md` - System architecture diagrams
- `README.md` - Project overview
- `LICENSE` - License information
