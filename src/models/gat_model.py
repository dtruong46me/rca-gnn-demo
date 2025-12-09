"""
GAT (Graph Attention Network) model for Root Cause Analysis.
Implements a 3-class classifier: 0=normal, 1=victim, 2=root cause.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv

from ..config import (
    DEFAULT_HIDDEN_CHANNELS,
    DEFAULT_GAT_HEADS_1,
    DEFAULT_GAT_HEADS_2,
    DEFAULT_OUTPUT_CLASSES,
    DEFAULT_DROPOUT
)


class GAT_RCA(nn.Module):
    """
    Graph Attention Network for Root Cause Analysis.
    
    Architecture:
        - GAT Layer 1: Multi-head attention with concatenation
        - GAT Layer 2: Multi-head attention without concatenation
        - Linear output layer for classification
        - Dropout applied after each GAT layer
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = DEFAULT_HIDDEN_CHANNELS,
        heads1: int = DEFAULT_GAT_HEADS_1,
        heads2: int = DEFAULT_GAT_HEADS_2,
        out_channels: int = DEFAULT_OUTPUT_CLASSES,
        dropout: float = DEFAULT_DROPOUT
    ):
        """
        Initialize GAT-RCA model.
        
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            heads1: Number of attention heads in first layer
            heads2: Number of attention heads in second layer
            out_channels: Output dimension (number of classes)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads1,
            concat=True,
            dropout=dropout
        )
        
        self.gat2 = GATConv(
            hidden_channels * heads1,
            hidden_channels,
            heads=heads2,
            concat=False,
            dropout=dropout
        )
        
        self.lin = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Node feature matrix (num_nodes x in_channels)
            edge_index: Graph connectivity (2 x num_edges)
            
        Returns:
            Node classification logits (num_nodes x out_channels)
        """
        # First GAT layer with ELU activation
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        
        # Second GAT layer with ELU activation
        x = F.elu(self.gat2(x, edge_index))
        x = self.dropout(x)
        
        # Linear output layer
        out = self.lin(x)  # [num_nodes, out_channels]
        
        return out
