# main.py
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import config
from data_processor import TelcoGraphDataset
from model import RootCauseGNN, FocalLoss
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import os

def train():
    print(">>> Initializing Dataset...")
    # Mode 'train' sẽ thực hiện fit vectorizer
    dataset_handler = TelcoGraphDataset(mode='train')
    
    print(">>> Creating Time Windows (this may take a moment)...")
    # Chia ticket theo cửa sổ 30 phút
    all_windows = dataset_handler.create_time_windows(window_size_min=30)
    
    if len(all_windows) == 0:
        print("ERROR: No data windows created. Check dataset_tickets.csv content.")
        return

    # Split Train (80%) / Test (20%) theo thời gian
    split_idx = int(len(all_windows) * 0.8)
    train_windows = all_windows[:split_idx]
    test_windows = all_windows[split_idx:]
    
    print(f"Total Windows: {len(all_windows)}. Train: {len(train_windows)}, Test: {len(test_windows)}")
    
    print(">>> Converting Windows to Graph Data Objects...")
    train_data_list = [dataset_handler.df_to_graph_data(df) for df in train_windows]
    test_data_list = [dataset_handler.df_to_graph_data(df) for df in test_windows]
    
    # DataLoader
    train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data_list, batch_size=32, shuffle=False)

    # Setup Model
    sample_data = train_data_list[0]
    num_features = sample_data.num_features
    print(f"Model Input Features: {num_features}")
    
    model = RootCauseGNN(num_features, config.HIDDEN_CHANNELS, num_classes=1).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=5e-4)
    
    # Tính Positive Weight cho Loss (Xử lý mất cân bằng dữ liệu)
    total_pos = sum([d.y.sum().item() for d in train_data_list]) # type: ignore
    total_neg = sum([(d.y == 0).sum().item() for d in train_data_list]) # type: ignore
    pos_weight_val = total_neg / (total_pos + 1e-5)
    pos_weight = torch.tensor([pos_weight_val]).to(config.DEVICE)
    
    print(f"⚖️ Calculated Pos Weight: {pos_weight.item():.2f}")
    
    # Loss function cho Binary Classification với Logits
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Hoặc dùng Focal Loss
    # criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')

    print("\n--- START TRAINING ---")
    best_f1 = 0.0
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(config.DEVICE)
            optimizer.zero_grad()
            
            # Forward
            out = model(batch) 
            
            # Tính Loss (Squeeze để out có shape [N] khớp với batch.y)
            loss = criterion(out.squeeze(), batch.y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)

        # Validation mỗi 5 epoch
        if (epoch + 1) % 5 == 0:
            val_metrics = evaluate(model, test_loader)
            print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | "
                  f"Val Recall: {val_metrics['recall']:.2f} | "
                  f"Val Prec: {val_metrics['precision']:.2f} | "
                  f"Val F1: {val_metrics['f1']:.2f}")
            
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                torch.save(model.state_dict(), config.MODEL_PATH)
                print(f"   >>> New Best Model Saved (F1: {best_f1:.2f})")

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(config.DEVICE)
            out = model(batch)
            
            # Sigmoid để đưa về xác suất [0, 1]
            probs = torch.sigmoid(out.squeeze())
            
            # Threshold 0.5 (Có thể tune số này)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
    return {
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0)
    }

if __name__ == "__main__":
    # Kiểm tra xem đã có data chưa
    if not os.path.exists(config.TICKET_FILE):
        print("⚠️ Data files not found! Please run 'generate_data.py' first.")
    else:
        train()