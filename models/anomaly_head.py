#!/usr/bin/env python3
"""
Lightweight anomaly detection head trained on frozen backbone.
Takes penultimate features (32-dim) → anomaly score (scalar).
"""

import copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm


class AnomalyHead(nn.Module):
    """Small MLP: 32 → 16 → 1 anomaly score."""
    def __init__(self, input_dim=32, hidden_dim=16, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, penult_features):
        return self.net(penult_features).squeeze(-1)  # (B,)


def generate_training_outliers(known_features, known_labels, prototypes, config, device):
    """
    Generate synthetic outliers for anomaly head training.
    Same strategies as train_v2.py but adapted for offline feature collection.
    """
    B, D = known_features.shape
    K = prototypes.shape[0]
    n = config.OUTLIER_COUNT

    proto = prototypes.to(device)
    outliers = []

    half = n // 2
    # Inter-class prototype interpolation
    for _ in range(half):
        i, j = random.sample(range(K), 2)
        alpha = random.random()
        interp = alpha * proto[i] + (1 - alpha) * proto[j]
        noise = config.OUTLIER_NOISE * torch.randn(D, device=device)
        outliers.append(interp + noise)

    # Perturb real features
    for _ in range(n - half):
        idx = random.randint(0, B - 1)
        feat = known_features[idx].detach()
        noise = config.OUTLIER_NOISE * 1.5 * torch.randn(D, device=device)
        outliers.append(feat + noise)

    return torch.stack(outliers)


def collect_features(model, dataloader, device):
    """Collect penultimate features and labels from known training samples."""
    model.eval()
    all_features = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="  Collecting features"):
            labels = batch["labels"].to(device)
            fused = model.embedding(
                batch["opc_indices"].to(device),
                batch["param_indices"].to(device),
                batch["stats"].to(device),
                model.atom_registry,
            )
            output = model(fused, batch["edge_index"].to(device),
                           batch["edge_attr"].to(device), batch["mask"].to(device))
            all_features.append(output["penult_features"].cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def train_anomaly_head(model, anomaly_head, train_loader, prototypes, config, device,
                        epochs=20, lr=1e-3):
    """
    Train anomaly head on frozen backbone.
    Known samples → anomaly score 0. Outliers → anomaly score 1.
    """
    # Freeze backbone
    for p in model.parameters():
        p.requires_grad = False

    anomaly_head = anomaly_head.to(device)
    anomaly_head.train()

    optimizer = AdamW(anomaly_head.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    # Collect all known features once
    print("  Collecting training features...")
    known_feats, known_labels = collect_features(model, train_loader, device)
    print(f"  Collected {known_feats.shape[0]} features, dim={known_feats.shape[1]}")

    # Class-balanced weights for normal samples: up-weight small classes (RE/BN)
    # so the head learns to score their normal samples low instead of uniformly high.
    from collections import Counter
    class_counts = Counter(known_labels.tolist())
    n_total = len(known_labels)
    n_classes = config.NUM_KNOWN
    class_weight = {c: n_total / (n_classes * class_counts[c]) for c in class_counts}
    print(f"  Class-balanced weights: { {c: round(class_weight[c], 2) for c in sorted(class_weight)} }")

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        # Shuffle and batch
        indices = torch.randperm(known_feats.size(0))
        batch_size = config.BATCH_SIZE * 4  # can use larger batches for feature vectors

        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            idx = indices[start:end]

            feats = known_feats[idx].to(device)  # (B, 32)
            B = feats.size(0)

            # Generate outliers
            outliers = generate_training_outliers(
                feats, known_labels[idx], prototypes, config, device)

            # Forward
            normal_scores = anomaly_head(feats)    # should be low
            outlier_scores = anomaly_head(outliers)  # should be high

            # BCE loss (class-balanced for normal samples)
            normal_targets = torch.zeros(B, device=device)
            outlier_targets = torch.ones(outliers.size(0), device=device)

            w = torch.tensor([class_weight[l.item()] for l in known_labels[idx]],
                             device=device, dtype=torch.float32)
            loss_normal = (criterion(normal_scores, normal_targets) * w).mean()
            loss_outlier = criterion(outlier_scores, outlier_targets).mean()
            loss = loss_normal + loss_outlier

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        print(f"  Anomaly Epoch {epoch+1:3d}/{epochs} | Loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = copy.deepcopy(anomaly_head.state_dict())

    # Restore best
    if best_state is not None:
        anomaly_head.load_state_dict(best_state)

    # Unfreeze backbone
    for p in model.parameters():
        p.requires_grad = True

    return anomaly_head
