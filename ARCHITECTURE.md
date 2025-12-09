# RCA-GNN Architecture Overview

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      RCA-GNN SYSTEM                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    1. DATA GENERATION                            │
├─────────────────────────────────────────────────────────────────┤
│  generators/                                                     │
│   ├── device_generator.py      → Generate network devices       │
│   ├── topology_generator.py    → Build network topology         │
│   ├── event_generator.py       → Create system events           │
│   ├── incident_generator.py    → Generate incidents + labels    │
│   └── customer_service_generator.py → Customer/service data     │
│                                                                  │
│  generate_samples_refactored.py → Orchestrates all generators   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      2. DATA STORAGE                             │
├─────────────────────────────────────────────────────────────────┤
│  CSV Files (data/samples/)                                       │
│   ├── devices.csv          ─┐                                   │
│   ├── edges.csv            ─┤                                   │
│   ├── events.csv           ─┼→ Input to training/inference      │
│   ├── incidents.csv        ─┤                                   │
│   └── node_labels.csv      ─┘                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   3. DATA LOADING & PREP                         │
├─────────────────────────────────────────────────────────────────┤
│  data/data_loader.py                                             │
│   ├── load_csvs()          → Read CSV files                     │
│   ├── preprocess_timestamps() → Parse datetime                  │
│   └── build_device_index() → Create device mapping              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  4. FEATURE ENGINEERING                          │
├─────────────────────────────────────────────────────────────────┤
│  features/feature_engineering.py                                 │
│   ├── Static Features                                            │
│   │   ├── fit_static_encoders()     → OneHot for vendor/layer   │
│   │   └── build_static_feature_matrix() → Encode features       │
│   │                                                              │
│   ├── Dynamic Features                                           │
│   │   └── aggregate_events_for_window() → Event counts          │
│   │                                                              │
│   ├── Topological Features                                       │
│   │   └── compute_degree_features() → Node degrees              │
│   │                                                              │
│   └── build_combined_features() → Concatenate all features      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    5. GRAPH CONSTRUCTION                         │
├─────────────────────────────────────────────────────────────────┤
│  graph/graph_builder.py                                          │
│   ├── build_topology_graph() → NetworkX graph                   │
│   ├── build_edge_index()    → PyG edge format                   │
│   └── build_samples()       → Create PyG Data objects           │
│                                                                  │
│  Output: List[Data(x, edge_index, y)]                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────┴─────────┐
                    │                   │
          ┌─────────▼─────────┐   ┌────▼──────────┐
          │  6. TRAINING      │   │ 7. INFERENCE  │
          └───────────────────┘   └───────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       6. TRAINING PATH                           │
├─────────────────────────────────────────────────────────────────┤
│  models/gat_model.py                                             │
│   └── GAT_RCA(in_channels, hidden, heads, dropout)              │
│       ├── GAT Layer 1 (multi-head attention)                    │
│       ├── GAT Layer 2 (multi-head attention)                    │
│       └── Linear classifier → 3 classes                          │
│                                                                  │
│  train/trainer.py                                                │
│   ├── train_model()                                              │
│   │   ├── Train/test split                                      │
│   │   ├── Training loop (epochs)                                │
│   │   ├── Backpropagation                                       │
│   │   └── Best model selection                                  │
│   │                                                              │
│   └── evaluate_model()                                           │
│       ├── Top-1 root accuracy                                    │
│       ├── Top-3 root accuracy                                    │
│       └── Per-class accuracy                                     │
│                                                                  │
│  Output: Trained model → saved to output/                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      7. INFERENCE PATH                           │
├─────────────────────────────────────────────────────────────────┤
│  inference/predictor.py                                          │
│   ├── infer_from_events()                                        │
│   │   ├── Aggregate new event window                            │
│   │   ├── Build features                                         │
│   │   ├── Run model inference                                    │
│   │   └── Return top-k candidates                               │
│   │                                                              │
│   └── predict_root_causes()                                      │
│       └── Simple prediction from features                        │
│                                                                  │
│  Output: List of root cause candidates with probabilities       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    8. UTILITIES & CONFIG                         │
├─────────────────────────────────────────────────────────────────┤
│  config.py                                                       │
│   ├── Model hyperparameters                                      │
│   ├── Training parameters                                        │
│   ├── Data generation settings                                   │
│   └── File paths                                                 │
│                                                                  │
│  utils.py                                                        │
│   ├── save_model_and_metadata()                                  │
│   ├── load_model_state()                                         │
│   ├── get_device()                                               │
│   └── print_model_summary()                                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   9. MAIN ORCHESTRATOR                           │
├─────────────────────────────────────────────────────────────────┤
│  main_refactored.py                                              │
│   ├── train_mode()    → Complete training pipeline              │
│   └── infer_mode()    → Complete inference pipeline             │
│                                                                  │
│  CLI Arguments:                                                  │
│   --mode [train|infer]                                           │
│   --data_dir, --out_dir, --epochs, --lr, --window_mins, etc.   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
INPUT DATA
    │
    ├─→ devices.csv ──────────┐
    ├─→ edges.csv ────────────┤
    ├─→ events.csv ───────────┼─→ Data Loader
    ├─→ incidents.csv ────────┤
    └─→ node_labels.csv ──────┘
                              │
                              ▼
                    Feature Engineering
                    ┌─────────┴─────────┐
                    │                   │
              Static Features    Dynamic Features
              (vendor, layer)    (event counts)
                    │                   │
                    └────────┬──────────┘
                             │
                             ▼
                    Graph Construction
                    (NetworkX → PyG)
                             │
                             ▼
                    Per-Incident Samples
                    [Data(x, edge_index, y)]
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
           TRAINING                  INFERENCE
                │                         │
          GAT Model ←──────────────── Load Model
                │                         │
         Train Loop                  Predict
                │                         │
         Save Model                  Top-K Results
                │                         │
                ▼                         ▼
           output/                   Candidates
        gat_rca_state.pt          [device, prob]
```

## Module Dependencies

```
main_refactored.py
    ├── config.py
    ├── utils.py
    ├── data/
    │   └── data_loader.py
    ├── features/
    │   └── feature_engineering.py
    ├── graph/
    │   └── graph_builder.py
    ├── models/
    │   └── gat_model.py
    ├── train/
    │   └── trainer.py
    └── inference/
        └── predictor.py

graph_builder.py
    └── features/
        └── feature_engineering.py

trainer.py
    └── models/
        └── gat_model.py

predictor.py
    ├── models/
    │   └── gat_model.py
    ├── features/
    │   └── feature_engineering.py
    └── graph/
        └── graph_builder.py
```

## Class Hierarchy

```
torch.nn.Module
    └── GAT_RCA
        ├── GATConv (layer 1)
        ├── GATConv (layer 2)
        └── Linear (classifier)

torch_geometric.data.Data
    ├── x: Node features [num_nodes × feature_dim]
    ├── edge_index: Graph structure [2 × num_edges]
    └── y: Node labels [num_nodes]
```

## Label Schema

```
Node Labels (3 classes):
    0 → Normal (not affected by incident)
    1 → Victim (affected downstream from root)
    2 → Root Cause (origin of incident)
    
Generated via BFS from root cause device:
    Root → Label 2
    All downstream devices → Label 1
    All other devices → Label 0
```
