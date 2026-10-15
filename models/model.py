#!/usr/bin/env python3
"""
VulnDetectionModel: CNN-BiLSTM-Attention-GCN architecture.
Section 4.6 of the paper.

Architecture:
  Input → 1D CNN(128, k=3) → BiLSTM(64×2)                [Branch A]
  Input → 2-layer SparseGCN(204→128→64)                   [Branch B]
  Concat(A,B) → MultiHeadAttention(8) → FC(64→32→K)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Sparse GCN Layer  (manual implementation, no torch_geometric dependency)
# ==============================================================================

class SparseGCNLayer(nn.Module):
    """Graph Convolution Layer using sparse scatter operations."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, edge_index, edge_attr=None):
        N = x.size(0)
        if edge_index.numel() == 0:
            out = x @ self.weight
            if self.bias is not None:
                out = out + self.bias
            return F.relu(out)

        src, dst = edge_index[0], edge_index[1]
        self_loops = torch.arange(N, device=x.device)
        all_src = torch.cat([src, self_loops])
        all_dst = torch.cat([dst, self_loops])
        if edge_attr is not None:
            all_weights = torch.cat([edge_attr, torch.ones(N, device=x.device)])
        else:
            all_weights = torch.ones(all_src.size(0), device=x.device)

        degree = torch.zeros(N, device=x.device).scatter_add(0, all_dst, all_weights)
        degree = degree.clamp(min=1)
        deg_inv_sqrt = degree.pow(-0.5)
        norm_weights = all_weights * deg_inv_sqrt[all_dst] * deg_inv_sqrt[all_src]

        x_w = x @ self.weight
        messages = x_w[all_src] * norm_weights.unsqueeze(-1)
        out = torch.zeros(N, x_w.size(1), device=x.device).scatter_add(
            0, all_dst.unsqueeze(-1).expand_as(messages), messages)

        if self.bias is not None:
            out = out + self.bias
        return F.relu(out)


class SparseGCN(nn.Module):
    """Two-layer GCN for data-dependency modeling."""

    def __init__(self, in_dim, hidden_dim=128, out_dim=64, dropout=0.3):
        super().__init__()
        self.layer1 = SparseGCNLayer(in_dim, hidden_dim)
        self.layer2 = SparseGCNLayer(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.layer1(x, edge_index, edge_attr)
        x = self.dropout(x)
        x = self.layer2(x, edge_index, edge_attr)
        return x


# ==============================================================================
# Main Model
# ==============================================================================

class VulnDetectionModel(nn.Module):
    """CNN-BiLSTM-Attention-GCN model with external embedding."""

    def __init__(self, config, num_classes: int):
        super().__init__()
        self._config = config
        self.num_classes = num_classes
        self.embedding = None
        self.atom_registry = None

        Fdim = config.FUSED_DIM
        C = config.CONCAT_DIM

        # Branch A: CNN + BiLSTM
        self.cnn = nn.Conv1d(Fdim, config.CNN_FILTERS, kernel_size=config.CNN_KERNEL,
                             padding='same', bias=True)
        self.cnn_act = nn.ReLU()
        self.cnn_drop = nn.Dropout(0.3)
        self.bilstm = nn.LSTM(config.CNN_FILTERS, config.BILSTM_UNITS,
                              num_layers=1, bidirectional=True, batch_first=True)

        # Branch B: GCN
        self.gcn = SparseGCN(Fdim, config.GCN_HIDDEN, config.GCN_OUT, config.GCN_DROPOUT)

        # Multi-Head Attention
        self.attn = nn.MultiheadAttention(C, config.ATTENTION_HEADS,
                                          dropout=0.1, batch_first=True)

        # FC layers
        self.fc1 = nn.Linear(C, config.FC1_UNITS)
        self.fc1_drop = nn.Dropout(config.DROPOUT1)
        self.fc2 = nn.Linear(config.FC1_UNITS, config.FC2_UNITS)
        self.fc2_drop = nn.Dropout(config.DROPOUT2)
        self.out = nn.Linear(config.FC2_UNITS, num_classes)

    @property
    def config(self):
        return self._config

    def set_embedding(self, embedding_layer, atom_registry):
        self.embedding = embedding_layer
        self.atom_registry = atom_registry
        # Ensure embedding is on the same device as model parameters
        if next(self.parameters(), None) is not None:
            device = next(self.parameters()).device
            self.embedding = self.embedding.to(device)

    def forward(self, fused_features, edge_index, edge_attr, mask=None):
        B, L, Fdim = fused_features.shape

        # Branch A: CNN → BiLSTM
        cnn_out = self.cnn_act(self.cnn(fused_features.transpose(1, 2)))
        cnn_out = self.cnn_drop(cnn_out)
        lstm_in = cnn_out.transpose(1, 2)  # (B, L, 128)
        if mask is not None:
            lengths = mask.sum(dim=1).long().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                lstm_in, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.bilstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=L)
        else:
            lstm_out, _ = self.bilstm(lstm_in)

        # Branch B: GCN
        flat = fused_features.reshape(B * L, Fdim)
        gcn_out = self.gcn(flat, edge_index, edge_attr).reshape(B, L, -1)

        # Concat + Attention
        combined = torch.cat([lstm_out, gcn_out], dim=-1)
        kpm = ~mask if mask is not None else None
        attn_out, attn_w = self.attn(combined, combined, combined, key_padding_mask=kpm)

        # Pool
        if mask is not None:
            m = mask.unsqueeze(-1).float()
            ctx = (attn_out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            ctx = attn_out.mean(dim=1)

        # FC
        x = F.relu(self.fc1(ctx)); x = self.fc1_drop(x)
        x = F.relu(self.fc2(x)); penult = self.fc2_drop(x)
        logits = self.out(penult)

        # Average attention received by each position across all query positions
        pos_attn = attn_w.mean(dim=1)  # (B, L)
        return {"logits": logits, "penult_features": penult,
                "attention_weights": pos_attn}


# ==============================================================================
# Ablation Variant Builders
# ==============================================================================

def build_model_variant(config, num_classes, variant="full"):
    if variant == "full":
        return VulnDetectionModel(config, num_classes)

    if variant == "no_param":
        class M(VulnDetectionModel):
            def __init__(self, cfg, nc):
                cfg = _copy_config(cfg)
                cfg.USE_PARAM_FEATURES = False
                super().__init__(cfg, nc)
        return M(config, num_classes)

    if variant == "no_gnn":
        class M(VulnDetectionModel):
            def __init__(self, cfg, nc):
                cfg = _copy_config(cfg)
                cfg.USE_GCN_BRANCH = False
                super().__init__(cfg, nc)
            def forward(self, fused, ei, ea, mask=None):
                B, L, Fdim = fused.shape
                cnn_out = self.cnn_act(self.cnn(fused.transpose(1, 2)))
                cnn_out = self.cnn_drop(cnn_out)
                lstm_in = cnn_out.transpose(1, 2)
                if mask is not None:
                    lengths = mask.sum(dim=1).long().clamp(min=1)
                    p = nn.utils.rnn.pack_padded_sequence(lstm_in, lengths.cpu(), batch_first=True, enforce_sorted=False)
                    po, _ = self.bilstm(p)
                    lstm_out, _ = nn.utils.rnn.pad_packed_sequence(po, batch_first=True, total_length=L)
                else:
                    lstm_out, _ = self.bilstm(lstm_in)
                kpm = ~mask if mask is not None else None
                ao, aw = self.attn(lstm_out, lstm_out, lstm_out, key_padding_mask=kpm)
                if mask is not None:
                    m = mask.unsqueeze(-1).float()
                    ctx = (ao * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
                else:
                    ctx = ao.mean(dim=1)
                x = F.relu(self.fc1(ctx)); x = self.fc1_drop(x)
                x = F.relu(self.fc2(x)); penult = self.fc2_drop(x)
                logits = self.out(penult)
                return {"logits": logits, "penult_features": penult,
                        "attention_weights": aw.mean(dim=1)}
        return M(config, num_classes)

    if variant == "no_stats":
        class M(VulnDetectionModel):
            def __init__(self, cfg, nc):
                cfg = _copy_config(cfg)
                cfg.USE_STATS_FEATURES = False
                super().__init__(cfg, nc)
        return M(config, num_classes)

    if variant == "single_head":
        class M(VulnDetectionModel):
            def __init__(self, cfg, nc):
                cfg = _copy_config(cfg)
                cfg.ATTENTION_HEADS = 1
                super().__init__(cfg, nc)
        return M(config, num_classes)

    raise ValueError(f"Unknown variant: {variant}")


def _copy_config(cfg):
    """Deep copy config to avoid mutating the original across ablation variants."""
    import copy
    return copy.deepcopy(cfg)
