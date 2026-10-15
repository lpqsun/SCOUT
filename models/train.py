#!/usr/bin/env python3
"""
Training loop with label-smoothing cross-entropy and prototype compactness regularization.
Section 4.1.5 of the paper.
"""

import os
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


def compute_loss(logits, labels, penult_features, class_prototypes, config):
    """
    Compute the combined loss.
    L = LabelSmoothingCE + λ * PrototypeCompactness

    Args:
        logits: (B, K) raw logits
        labels: (B,) class indices
        penult_features: (B, D) penultimate layer features (32-dim)
        class_prototypes: (K, D) prototype vectors for each known class
        config: Config object
    Returns:
        total_loss, ce_loss, proto_loss
    """
    K = config.NUM_KNOWN
    B = logits.size(0)
    eps = config.LABEL_SMOOTHING
    device = logits.device

    # 1. Label Smoothing Cross-Entropy
    # Build smoothed targets: (B, K)
    with torch.no_grad():
        smooth_labels = torch.full((B, K), eps / (K - 1), device=device)
        smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - eps)

    log_probs = F.log_softmax(logits, dim=1)
    ce_loss = -(smooth_labels * log_probs).sum(dim=1).mean()

    # 2. Prototype Compactness Regularization
    proto_loss = torch.tensor(0.0, device=device)
    if class_prototypes is not None and config.PROTOTYPE_LAMBDA > 0:
        for k in range(K):
            mask = (labels == k)
            if mask.sum() > 0:
                class_feats = penult_features[mask]  # (N_k, D)
                diff = class_feats - class_prototypes[k].unsqueeze(0).to(device)
                proto_loss += (diff.pow(2)).sum()
        proto_loss = proto_loss / B

    total_loss = ce_loss + config.PROTOTYPE_LAMBDA * proto_loss
    return total_loss, ce_loss, proto_loss


def compute_accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean()


def _build_batch_fused(model, batch, device):
    """Build fused feature tensor from batch using the model's HybridEmbedding."""
    opc_indices = batch["opc_indices"].to(device)
    param_indices = batch["param_indices"].to(device)
    stats = batch["stats"].to(device)
    return model.embedding(opc_indices, param_indices, stats, model.atom_registry)


def train_epoch(model, dataloader, optimizer, config, prototypes, device="cpu"):
    """Single training epoch."""
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_proto = 0.0
    total_acc = 0.0
    n_batches = 0

    all_features = [[] for _ in range(config.NUM_KNOWN)]

    for batch in dataloader:
        labels = batch["labels"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        mask = batch["mask"].to(device)
        fused = _build_batch_fused(model, batch, device)

        output = model(fused, edge_index, edge_attr, mask)
        logits = output["logits"]
        penult = output["penult_features"]

        loss, ce, proto = compute_loss(logits, labels, penult, prototypes, config)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            acc = compute_accuracy(logits, labels)

        total_loss += loss.item()
        total_ce += ce.item()
        total_proto += proto.item()
        total_acc += acc.item()
        n_batches += 1

        # Collect features per class
        for k in range(config.NUM_KNOWN):
            mask_k = (labels == k)
            if mask_k.sum() > 0:
                all_features[k].append(penult[mask_k].detach())

    avg_loss = total_loss / n_batches if n_batches > 0 else 0
    avg_ce = total_ce / n_batches if n_batches > 0 else 0
    avg_proto = total_proto / n_batches if n_batches > 0 else 0
    avg_acc = total_acc / n_batches if n_batches > 0 else 0

    # Update prototypes via EMA
    new_prototypes = prototypes.clone() if prototypes is not None else torch.zeros(config.NUM_KNOWN, config.FC2_UNITS)
    for k in range(config.NUM_KNOWN):
        if all_features[k]:
            cat_feats = torch.cat(all_features[k], dim=0)
            class_mean = cat_feats.mean(dim=0).cpu()
            if prototypes is not None and prototypes[k].sum() != 0:
                new_prototypes[k] = config.PROTOTYPE_EMA_ALPHA * prototypes[k] + (1 - config.PROTOTYPE_EMA_ALPHA) * class_mean
            else:
                new_prototypes[k] = class_mean

    return avg_loss, avg_ce, avg_proto, avg_acc, new_prototypes


@torch.no_grad()
def validate(model, dataloader, config, prototypes, device="cpu"):
    """Validation epoch."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for batch in dataloader:
        labels = batch["labels"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        mask = batch["mask"].to(device)
        fused = _build_batch_fused(model, batch, device)

        output = model(fused, edge_index, edge_attr, mask)
        logits = output["logits"]
        penult = output["penult_features"]

        loss, _, _ = compute_loss(logits, labels, penult, prototypes, config)
        acc = compute_accuracy(logits, labels)

        total_loss += loss.item()
        total_acc += acc.item()
        n_batches += 1

    avg_loss = total_loss / n_batches if n_batches > 0 else 0
    avg_acc = total_acc / n_batches if n_batches > 0 else 0
    return avg_loss, avg_acc


def train(config, model, train_loader, val_loader, seed: int = 42,
          device: str = "cpu"):
    """
    Full training loop with early stopping.

    Returns:
        best_model_state, training_history, final_prototypes
    """
    torch.manual_seed(seed)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE,
                      weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    prototypes = torch.zeros(config.NUM_KNOWN, config.FC2_UNITS)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(config.EPOCHS):
        train_loss, train_ce, train_proto, train_acc, prototypes = train_epoch(
            model, train_loader, optimizer, config, prototypes, device)

        val_loss, val_acc = validate(model, val_loader, config, prototypes, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

        print(f"  Epoch {epoch+1:3d}/{config.EPOCHS} | "
              f"Train L={train_loss:.3f} CE={train_ce:.3f} Proto={train_proto:.4f} Acc={train_acc:.3f} | "
              f"Val L={val_loss:.3f} Acc={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_state)

    # Compute final prototypes from all training data (exact mean, not EMA)
    print("  Computing final prototypes...")
    model.eval()
    final_prototypes = torch.zeros(config.NUM_KNOWN, config.FC2_UNITS)
    proto_counts = torch.zeros(config.NUM_KNOWN)
    with torch.no_grad():
        for batch in train_loader:
            labels = batch["labels"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device)
            mask = batch["mask"].to(device)
            fused = _build_batch_fused(model, batch, device)

            output = model(fused, edge_index, edge_attr, mask)
            penult = output["penult_features"].cpu()

            for k in range(config.NUM_KNOWN):
                mask_k = (labels.cpu() == k)
                n_k = mask_k.sum().item()
                if n_k > 0:
                    final_prototypes[k] += penult[mask_k].sum(dim=0)
                    proto_counts[k] += n_k

    for k in range(config.NUM_KNOWN):
        if proto_counts[k] > 0:
            final_prototypes[k] /= proto_counts[k]

    return best_state, history, final_prototypes
