"""
rca_gat_demo.py
- Preprocess CSVs -> build dataset (per-incident graph snapshots)
- Train simple GAT (3 classes: 0-normal,1-victim,2-root)
- Inference for new 5-min window events
"""

import os
import argparse
import json
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ---------------------------
# Utility functions
# ---------------------------
def load_csvs(data_dir="."):
    devices = pd.read_csv(os.path.join(data_dir, "devices.csv"))
    edges = pd.read_csv(os.path.join(data_dir, "edges.csv"))
    events = pd.read_csv(os.path.join(data_dir, "events.csv"))
    incidents = pd.read_csv(os.path.join(data_dir, "incidents.csv"))
    labels = pd.read_csv(os.path.join(data_dir, "node_labels.csv"))
    return devices, edges, events, incidents, labels

def build_device_index(devices_df):
    device_list = devices_df['device_id'].unique().tolist()
    device_list.sort()
    idx = {d:i for i,d in enumerate(device_list)}
    return device_list, idx

def build_topology_graph(edges_df, device_list):
    # undirected graph for message passing
    G = nx.Graph()
    G.add_nodes_from(device_list)
    for _, r in edges_df.iterrows():
        s = r['source']
        t = r['target']
        if s in device_list and t in device_list:
            G.add_edge(s, t, **{k:r[k] for k in r.index if k not in ['source','target']})
    return G

def compute_degree_features(G, device_list, device_index):
    deg = np.zeros((len(device_list),1), dtype=float)
    for d in device_list:
        deg[device_index[d],0] = G.degree(d)
    return deg

# Simple event aggregation per device for a time window
def aggregate_events_for_window(events_df, device_list, device_index, window_center_time, window_mins=5):
    start = window_center_time - timedelta(minutes=window_mins)
    end = window_center_time
    # ensure timestamp column is datetime
    if events_df['timestamp'].dtype == object:
        events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])
    subset = events_df[(events_df['timestamp'] >= start) & (events_df['timestamp'] <= end)]
    event_count = np.zeros((len(device_list),1), dtype=float)
    critical_count = np.zeros((len(device_list),1), dtype=float)
    for _, r in subset.iterrows():
        dev = r['device_id']
        if dev in device_index:
            i = device_index[dev]
            event_count[i,0] += 1
            if str(r.get('severity',"")).lower() == 'critical':
                critical_count[i,0] += 1
    return event_count, critical_count

# build node static features (vendor, layer) using one-hot encoders
def fit_static_encoders(devices_df):
    enc_vendor = OneHotEncoder(sparse=False, handle_unknown='ignore')
    enc_layer = OneHotEncoder(sparse=False, handle_unknown='ignore')
    enc_vendor.fit(devices_df[['vendor']])
    enc_layer.fit(devices_df[['layer']])
    return enc_vendor, enc_layer

def build_static_feature_matrix(devices_df, device_list, device_index, enc_vendor, enc_layer):
    n = len(device_list)
    # prepare arrays by index order
    vendor_vals = []
    layer_vals = []
    for d in device_list:
        row = devices_df[devices_df['device_id'] == d]
        if row.shape[0] == 0:
            vendor_vals.append(['UNKNOWN'])
            layer_vals.append(['UNKNOWN'])
        else:
            vendor_vals.append([row.iloc[0]['vendor']])
            layer_vals.append([row.iloc[0]['layer']])
    vendor_mat = enc_vendor.transform(vendor_vals)  # n x vdim
    layer_mat = enc_layer.transform(layer_vals)    # n x ldim
    return vendor_mat, layer_mat

# ---------------------------
# Build per-incident samples
# ---------------------------
def build_samples(devices, edges, events, incidents, labels, window_mins=5):
    # device index & topology
    device_list, device_index = build_device_index(devices)
    G = build_topology_graph(edges, device_list)
    # encoders
    enc_vendor, enc_layer = fit_static_encoders(devices)
    vendor_mat, layer_mat = build_static_feature_matrix(devices, device_list, device_index, enc_vendor, enc_layer)
    degree_feat = compute_degree_features(G, device_list, device_index)  # n x 1

    # prepare adjacency (edge_index) as torch tensor once (use full device graph)
    edge_index_list = []
    for u,v in G.edges():
        edge_index_list.append((device_index[u], device_index[v]))
        edge_index_list.append((device_index[v], device_index[u]))
    if len(edge_index_list)==0:
        edge_index = torch.empty((2,0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()

    # pre-merge labels by incident: labels DF has rows (incident_id, device_id, label)
    labels_grouped = labels.groupby('incident_id')

    samples = []
    incident_ids = []
    for _, inc in incidents.iterrows():
        inc_id = inc['incident_id']
        # incident time
        try:
            t = pd.to_datetime(inc['timestamp'])
        except:
            t = pd.to_datetime(datetime.now())
        # aggregated event features
        ev_count, crit_count = aggregate_events_for_window(events, device_list, device_index, t, window_mins=window_mins)
        # combine static + topo + dynamic
        # vendor_mat (n x vdim), layer_mat (n x ldim), degree_feat (n x1), ev_count, crit_count
        X = np.concatenate([vendor_mat, layer_mat, degree_feat, ev_count, crit_count], axis=1) # type: ignore
        X = torch.tensor(X, dtype=torch.float)

        # labels: if labels df contains this incident, use; otherwise skip
        if inc_id in labels_grouped.groups:
            lab_df = labels_grouped.get_group(inc_id)
            y = np.zeros(len(device_list), dtype=np.int64)
            for _, r in lab_df.iterrows():
                d = r['device_id']
                if d in device_index:
                    y[device_index[d]] = int(r['label'])
        else:
            # if no labels for this incident, skip
            continue
        y = torch.tensor(y, dtype=torch.long)  # N

        data = Data(x=X, edge_index=edge_index, y=y)
        samples.append(data)
        incident_ids.append(inc_id)

    return samples, device_list, device_index, edge_index, incident_ids

# ---------------------------
# GAT Model (3-class)
# ---------------------------
class GAT_RCA(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, heads1=4, heads2=2, out_channels=3, dropout=0.2):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads1, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_channels*heads1, hidden_channels, heads=heads2, concat=False, dropout=dropout)
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.gat2(x, edge_index))
        x = self.dropout(x)
        out = self.lin(x)  # [N, out_channels]
        return out

# ---------------------------
# Training & evaluation
# ---------------------------
def evaluate_model(model, samples, device='cpu'):
    model.eval()
    total = len(samples)
    correct_root_top1 = 0
    top3_hit = 0
    perclass_acc = {0:0,1:0,2:0}
    counts = {0:0,1:0,2:0}

    with torch.no_grad():
        for data in samples:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            y = data.y.to(device)  # N
            logits = model(x, edge_index)  # N x 3
            probs = F.softmax(logits, dim=1).cpu().numpy()  # N x 3
            # predicted root candidate: argmax over prob[:,2]
            root_scores = probs[:,2]
            pred_root_idx = int(root_scores.argmax())
            # true root is node id where y==2; assuming single root per incident
            true_roots = (y.cpu().numpy() == 2).nonzero()[0]
            true_root_idx = int(true_roots[0]) if len(true_roots)>0 else -1
            if pred_root_idx == true_root_idx:
                correct_root_top1 += 1
            # top3 by class-2 score
            top3 = list(np.argsort(-root_scores)[:3])
            if true_root_idx in top3:
                top3_hit += 1
            # per-class accuracy
            pred_labels = logits.argmax(dim=1).cpu().numpy()
            for c in [0,1,2]:
                counts[c] += (y.cpu().numpy() == c).sum()
                perclass_acc[c] += ((pred_labels == c) & (y.cpu().numpy() == c)).sum()

    # compute rates
    top1_rate = correct_root_top1 / max(1, total)
    top3_rate = top3_hit / max(1, total)
    perclass_acc_final = {c: (perclass_acc[c]/counts[c] if counts[c]>0 else None) for c in perclass_acc}
    return {"top1": top1_rate, "top3": top3_rate, "perclass_acc": perclass_acc_final, "num_samples": total}

def train_model(samples, in_dim, epochs=30, lr=1e-3, device='cpu'):
    # train/test split by samples
    train_idx, test_idx = train_test_split(list(range(len(samples))), test_size=0.2, random_state=42)
    train_samples = [samples[i] for i in train_idx]
    test_samples = [samples[i] for i in test_idx]

    model = GAT_RCA(in_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()  # per-node

    best_top1 = -1.0
    best_state = None

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for data in train_samples:
            optimizer.zero_grad()
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            y = data.y.to(device)  # N
            logits = model(x, edge_index)  # N x 3
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(train_samples))
        # eval
        stats = evaluate_model(model, test_samples, device=device)
        print(f"Epoch {epoch:02d} loss={avg_loss:.4f} top1={stats['top1']:.3f} top3={stats['top3']:.3f}")
        # save best
        if stats['top1'] > best_top1:
            best_top1 = stats['top1']
            best_state = model.state_dict()

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"train_idx":train_idx, "test_idx":test_idx, "metrics": stats} # type: ignore

# ---------------------------
# Inference function for new events window
# ---------------------------
def infer_from_events(model, devices_df, edges_df, events_df, device_list, device_index, edge_index, enc_vendor, enc_layer, window_center_time, window_mins=5, topk=3, device='cpu'):
    # compute aggregated event features for the window
    ev_count, crit_count = aggregate_events_for_window(events_df, device_list, device_index, window_center_time, window_mins=window_mins)
    vendor_mat, layer_mat = build_static_feature_matrix(devices_df, device_list, device_index, enc_vendor, enc_layer)
    degree_feat = compute_degree_features(build_topology_graph(edges_df, device_list), device_list, device_index)
    X = np.concatenate([vendor_mat, layer_mat, degree_feat, ev_count, crit_count], axis=1)
    X = torch.tensor(X, dtype=torch.float).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(X, edge_index.to(device))  # N x 3
        probs = F.softmax(logits, dim=1).cpu().numpy()  # N x 3
    root_probs = probs[:,2]
    top_idx = list(np.argsort(-root_probs)[:topk])
    results = []
    for idx in top_idx:
        results.append({"device_id": device_list[idx], "root_prob": float(root_probs[idx])})
    return results, probs

# ---------------------------
# Main CLI
# ---------------------------
def main(args):
    devices, edges, events, incidents, labels = load_csvs(args.data_dir)
    # Ensure events timestamps parsed:
    events['timestamp'] = pd.to_datetime(events['timestamp'])
    incidents['timestamp'] = pd.to_datetime(incidents['timestamp'])

    # build samples
    print("Building samples...")
    samples, device_list, device_index, edge_index, incident_ids = build_samples(devices, edges, events, incidents, labels, window_mins=args.window_mins)
    print(f"Built {len(samples)} samples; num_devices={len(device_list)}")

    if args.mode == 'train':
        in_dim = samples[0].x.shape[1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training device:", device)
        model, info = train_model(samples, in_dim, epochs=args.epochs, lr=args.lr, device=device) # type: ignore
        print("Training completed. Evaluation metrics on test set:")
        print(info['metrics'])

        # save model and metadata
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.out_dir, "gat_rca_state.pt"))
        meta = {"device_list": device_list, "device_index": device_index}
        with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
            json.dump(meta, f)
        print("Saved model and meta to", args.out_dir)
    elif args.mode == 'infer':
        # load model
        in_dim = samples[0].x.shape[1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GAT_RCA(in_dim).to(device)
        model.load_state_dict(torch.load(os.path.join(args.out_dir, "gat_rca_state.pt"), map_location=device))
        # prepare encoders for static features
        enc_vendor, enc_layer = fit_static_encoders(devices)
        # inference on a chosen window (last incident or now)
        t = pd.to_datetime(args.infer_time) if args.infer_time else (pd.to_datetime("now"))
        results, probs = infer_from_events(model, devices, edges, events, device_list, device_index, edge_index, enc_vendor, enc_layer, t, window_mins=args.window_mins, topk=args.topk, device=device) # type: ignore
        print("Top candidates:", results)
    else:
        print("Unknown mode. Use --mode train or --mode infer")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".", help="directory with CSV files")
    parser.add_argument("--out_dir", type=str, default="./output", help="where to save model and meta")
    parser.add_argument("--mode", type=str, default="train", choices=["train","infer"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window_mins", type=int, default=5)
    parser.add_argument("--infer_time", type=str, default=None, help="ISO timestamp for inference (optional)")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()
    main(args)
