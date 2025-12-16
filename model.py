# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import config  # <--- Đã thêm dòng này

# --- THÊM CLASS NÀY ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: Logits (chưa qua sigmoid)
        # targets: labels (0 hoặc 1)
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss) # pt là xác suất dự đoán đúng
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        

class RootCauseGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes=1):
        super(RootCauseGNN, self).__init__()
        
        # Layer 1: GAT Conv
        # heads=4 giúp model học được nhiều mối quan hệ khác nhau
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=4, dropout=config.DROPOUT)
        
        # Layer 2: GAT Conv
        # Input dim = hidden_channels * heads
        self.conv2 = GATConv(hidden_channels * 4, hidden_channels, heads=2, dropout=config.DROPOUT)
        
        # Layer 3: Output Layer
        # Trả về 1 giá trị duy nhất (Logit) cho mỗi node để dùng BCEWithLogitsLoss
        self.conv3 = GATConv(hidden_channels * 2, num_classes, heads=1, concat=False, dropout=config.DROPOUT)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=config.DROPOUT, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=config.DROPOUT, training=self.training)
        
        # Layer 3
        x = self.conv3(x, edge_index)
        
        # Trả về Logits (chưa qua Sigmoid)
        return x