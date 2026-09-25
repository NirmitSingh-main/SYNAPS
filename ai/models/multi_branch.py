"""
Multi-Branch Signal Classification Architecture for SYNAPS.
Combines:
- Branch A: Multi-scale Strided CNN (local waveform & transition features)
- Branch B: Transformer Encoder with Dual Statistical Pooling (temporal dynamics & token variance)
- Branch C: Constellation GNN with Exact Point Subsampling (topological graph over 2D constellation state space)
- Multi-Branch Fusion Head: Learnable fusion with BatchNorm and Dropout regularization
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstellationGNN(nn.Module):
    """
    Constellation Graph Neural Network (Pure PyTorch implementation).
    Constructs a topological graph over constellation points / token embeddings:
    - Nodes: N constellation points with features [I, Q, mag, diff_cos, diff_sin]
    - Edges: Dynamic adjacency based on pairwise 2D Euclidean distance in (I, Q) space + temporal transitions
    - Message Passing: 2-layer Graph Convolution (GCN) via degree-normalized graph Laplacian
    """

    def __init__(
        self,
        node_in_features: int = 5,
        hidden_dim: int = 32,
        out_dim: int = 64,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.node_proj = nn.Linear(node_in_features, hidden_dim)

        # GCN Layer 1
        self.gc1_weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.gc1_bn = nn.BatchNorm1d(hidden_dim)

        # GCN Layer 2
        self.gc2_weight = nn.Parameter(torch.Tensor(hidden_dim, out_dim))
        self.gc2_bn = nn.BatchNorm1d(out_dim)

        self.dropout = nn.Dropout(dropout)

        # Initialization
        nn.init.xavier_uniform_(self.gc1_weight)
        nn.init.xavier_uniform_(self.gc2_weight)

    def compute_adjacency(self, node_feats: torch.Tensor) -> torch.Tensor:
        """
        Compute degree-normalized adjacency matrix A_hat = D^{-1/2} (A + I) D^{-1/2}
        node_feats: (batch, N, C) where C >= 2 (channel 0=I, channel 1=Q)
        """
        batch_size, num_nodes, _ = node_feats.shape
        iq = node_feats[:, :, :2]  # (batch, N, 2)

        # Pairwise squared Euclidean distance in normalized (I, Q) constellation plane
        dist_sq = torch.cdist(iq, iq, p=2.0) ** 2

        # Gaussian RBF kernel adjacency (sigma parameter tuned for normalized constellation radius)
        gamma = 2.0
        A = torch.exp(-gamma * dist_sq)  # (batch, N, N)

        # Add temporal transition edges (connect consecutive tokens)
        eye = torch.eye(num_nodes, device=node_feats.device).unsqueeze(0)
        band = torch.diag(torch.ones(num_nodes - 1, device=node_feats.device), diagonal=1).unsqueeze(0)
        band = band + band.transpose(1, 2)
        A = A + 0.5 * band + eye

        # Degree normalization: D^{-1/2} A D^{-1/2}
        deg = A.sum(dim=-1, keepdim=True)  # (batch, N, 1)
        deg_inv_sqrt = torch.pow(torch.clamp(deg, min=1e-6), -0.5)
        A_norm = deg_inv_sqrt * A * deg_inv_sqrt.transpose(1, 2)
        return A_norm

    def forward(self, x_nodes: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x_nodes: Tensor of shape (batch, N, C_in)
        Returns:
            graph_embedding: Tensor of shape (batch, out_dim)
        """
        B, N, C = x_nodes.shape
        A_norm = self.compute_adjacency(x_nodes)  # (B, N, N)

        # Linear projection to hidden dimension
        h = self.node_proj(x_nodes)  # (B, N, hidden_dim)

        # GCN Layer 1: A_norm @ h @ W1
        h = torch.bmm(A_norm, h)
        h = torch.matmul(h, self.gc1_weight)
        h = self.gc1_bn(h.transpose(1, 2)).transpose(1, 2)
        h = F.gelu(h)
        h = self.dropout(h)

        # GCN Layer 2: A_norm @ h @ W2
        h = torch.bmm(A_norm, h)
        h = torch.matmul(h, self.gc2_weight)
        h = self.gc2_bn(h.transpose(1, 2)).transpose(1, 2)
        h = F.gelu(h)
        h = self.dropout(h)

        # Dual graph pooling (Mean + Max) to capture both cluster centroid and extreme constellation states
        h_mean = h.mean(dim=1)
        h_max, _ = h.max(dim=1)
        h_graph = (h_mean + h_max) * 0.5
        return h_graph


class MultiBranchSignalClassifier(nn.Module):
    """
    Multi-Branch Signal Classifier:
    - Branch A: CNN (Local waveform representation)
    - Branch B: Transformer with Dual Statistical Pooling (Temporal sequence & token variance)
    - Branch C: Constellation GNN with Exact Subsampling (2D Phase-space topology)
    - Fusion Head: Learnable MLP with BatchNorm, GELU, and Dropout
    """

    def __init__(
        self,
        input_channels: int = 5,
        num_classes: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        transformer_layers: int = 2,
        dropout: float = 0.35,
        num_tokens: int = 256,
        num_gnn_nodes: int = 64,
        use_gnn: bool = True,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_tokens = num_tokens
        self.num_gnn_nodes = num_gnn_nodes
        self.use_gnn = use_gnn

        # Branch A: Multi-scale Strided Conv1D
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=31, stride=4, padding=15),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, d_model, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.token_pool = nn.AdaptiveAvgPool1d(num_tokens)
        self.cnn_pool = nn.AdaptiveAvgPool1d(1)

        # Branch B: Transformer Encoder with Statistical Pooling Projection
        self.pos_emb = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
        self.transformer_norm = nn.LayerNorm(d_model)
        # Projects concatenated [mean, std] (2 * d_model) -> d_model
        self.trans_stat_proj = nn.Linear(d_model * 2, d_model)

        # Branch C: Constellation GNN
        if self.use_gnn:
            self.gnn = ConstellationGNN(
                node_in_features=input_channels,
                hidden_dim=32,
                out_dim=d_model,
                dropout=dropout,
            )
            fused_dim = d_model * 3  # CNN (64) + Transformer Stat (64) + GNN (64) = 192
        else:
            self.gnn = None
            fused_dim = d_model * 2  # CNN (64) + Transformer Stat (64) = 128

        # Feature Fusion & Classifier Head
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            x: Tensor with shape (batch, input_channels, N) e.g. (batch, 5, 9600)
        Returns:
            logits: Tensor with shape (batch, num_classes)
        """
        if x.ndim == 3 and x.shape[1] != self.input_channels and x.shape[2] == self.input_channels:
            x = x.transpose(1, 2)

        if x.ndim != 3 or x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected input of shape (batch, {self.input_channels}, samples), got {x.shape}"
            )

        B, C, N = x.shape

        # 1. Branch A: CNN feature representation
        feat_map = self.cnn(x)  # (B, d_model, N')
        h_cnn = self.cnn_pool(feat_map).squeeze(-1)  # (B, d_model)

        # 2. Branch B: Transformer with Dual Statistical Pooling (Mean + Std)
        tokens = self.token_pool(feat_map).transpose(1, 2)  # (B, num_tokens, d_model)
        tokens = tokens + self.pos_emb[:, :tokens.shape[1], :]
        h_trans_seq = self.transformer(tokens)  # (B, num_tokens, d_model)
        h_trans_norm = self.transformer_norm(h_trans_seq)
        h_trans_mean = h_trans_norm.mean(dim=1)  # (B, d_model)
        h_trans_std = h_trans_norm.std(dim=1)   # (B, d_model) - preserves multi-modal amplitude spread!
        h_trans = F.gelu(self.trans_stat_proj(torch.cat([h_trans_mean, h_trans_std], dim=-1)))  # (B, d_model)

        # 3. Branch C: Constellation GNN with Exact Point Subsampling (No low-pass smoothing)
        if self.use_gnn:
            step = max(1, N // self.num_gnn_nodes)
            gnn_nodes = x[:, :, ::step][:, :, :self.num_gnn_nodes].transpose(1, 2)  # (B, num_gnn_nodes, C)
            h_gnn = self.gnn(gnn_nodes)  # (B, d_model)
            h_fused = torch.cat([h_cnn, h_trans, h_gnn], dim=-1)  # (B, 3 * d_model)
        else:
            h_fused = torch.cat([h_cnn, h_trans], dim=-1)  # (B, 2 * d_model)

        # 4. Feature Fusion & Classification
        logits = self.fusion_head(h_fused)  # (B, num_classes)
        return logits
