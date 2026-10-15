#!/usr/bin/env python3
"""
Improved training with contrastive prototype loss and outlier exposure.
Section 5.4 of the paper — enhancements.
"""

import copy, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


def compute_loss_v2(logits, labels, penult_features, class_prototypes, config):
    """
    Combined loss:
      L = LabelSmoothingCE + λ_proto * ProtoCompactness + λ_rep * Repulsion
    """
    K = config.NUM_KNOWN
    B = logits.size(0)
    eps = config.LABEL_SMOOTHING
    device = logits.device

    # 1. Label Smoothing Cross-Entropy
    with torch.no_grad():
        smooth_labels = torch.full((B, K), eps / (K - 1), device=device)
        smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - eps)

    log_probs = F.log_softmax(logits, dim=1)
    ce_loss = -(smooth_labels * log_probs).sum(dim=1).mean()

    # 2. Prototype Compactness (intra-class attraction)
    proto_loss = torch.tensor(0.0, device=device)
    if class_prototypes is not None and config.PROTOTYPE_LAMBDA > 0:
        for k in range(K):
            mask = (labels == k)
            if mask.sum() > 0:
                class_feats = penult_features[mask]
                diff = class_feats - class_prototypes[k].unsqueeze(0).to(device)
                proto_loss += (diff.pow(2)).sum()
        proto_loss = proto_loss / B

    # 3. Inter-class Repulsion (push prototypes apart)
    repulsion_loss = torch.tensor(0.0, device=device)
    if class_prototypes is not None and config.REPULSION_LAMBDA > 0:
        proto_cpu = class_prototypes.to(device) if class_prototypes.device != device else class_prototypes
        n_pairs = 0
        for i in range(K):
            for j in range(i + 1, K):
                dist_ij = (proto_cpu[i] - proto_cpu[j]).pow(2).sum().sqrt()
                repulsion_loss += F.relu(config.REPULSION_MARGIN - dist_ij)
                n_pairs += 1
        if n_pairs > 0:
            repulsion_loss = repulsion_loss / n_pairs

    total_loss = ce_loss + config.PROTOTYPE_LAMBDA * proto_loss + config.REPULSION_LAMBDA * repulsion_loss
    return total_loss, ce_loss, proto_loss, repulsion_loss


def generate_outliers(penult_features, labels, prototypes, config, device):
    """
    Generate synthetic outlier features to teach the model to reject unknowns.

    Two strategies:
      1. Inter-class interpolation: α*p_i + (1-α)*p_j + noise
      2. Real feature perturbation: real_f + stronger noise
    """
    B, D = penult_features.shape
    K = prototypes.shape[0]
    n = config.OUTLIER_COUNT

    outliers = []
    proto_cpu = prototypes.to(device) if prototypes.device != device else prototypes

    half = n // 2

    # Strategy 1: Inter-class prototype interpolation
    for _ in range(half):
        i, j = random.sample(range(K), 2)
        alpha = random.random()
        interp = alpha * proto_cpu[i] + (1 - alpha) * proto_cpu[j]
        noise = config.OUTLIER_NOISE * torch.randn(D, device=device)
        outliers.append(interp + noise)

    # Strategy 2: Perturb real features
    for _ in range(n - half):
        idx = random.randint(0, B - 1)
        feat = penult_features[idx].detach()
        noise = config.OUTLIER_NOISE * 1.5 * torch.randn(D, device=device)
        outliers.append(feat + noise)

    return torch.stack(outliers)


def compute_outlier_loss(outlier_features, model_out_layer, class_prototypes, config, device):
    """
    Teach the model to be uncertain about outliers.
    Two complementary losses:
      1. Entropy maximization: push outlier predictions toward uniform distribution
      2. Prototype repulsion: push outliers away from ALL known prototypes
    """
    K = config.NUM_KNOWN
    n_out = outlier_features.size(0)

    # 1. Entropy maximization: make softmax output uniform
    logits = model_out_layer(outlier_features)
    probs = F.softmax(logits, dim=1)
    # Target: uniform distribution over K classes
    target = torch.ones(n_out, K, device=device) / K
    entropy_loss = F.kl_div(probs.log(), target, reduction='batchmean')

    # 2. Prototype repulsion: push outliers away from all prototypes
    proto = class_prototypes.to(device) if class_prototypes.device != device else class_prototypes
    dists = torch.cdist(outlier_features, proto.unsqueeze(0).expand(n_out, -1, -1).squeeze(0) if proto.dim() == 2 else proto)
    # dists: (n_out, K)
    min_dist, _ = dists.min(dim=1)  # distance to nearest prototype
    proto_loss = F.relu(config.OUTLIER_PROTO_MARGIN - min_dist).mean()

    return entropy_loss + config.OUTLIER_PROTO_LAMBDA * proto_loss


def compute_accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean()


def _build_batch_fused(model, batch, device):
    opc_indices = batch["opc_indices"].to(device)
    param_indices = batch["param_indices"].to(device)
    stats = batch["stats"].to(device)
    return model.embedding(opc_indices, param_indices, stats, model.atom_registry)


def train_epoch_v2(model, dataloader, optimizer, config, prototypes, device="cpu"):
    """Single training epoch with outlier exposure."""
    model.train()
    total_loss = total_ce = total_proto = total_rep = total_out = 0.0
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

        # Core loss: CE + prototype compactness + repulsion
        loss, ce, proto, rep = compute_loss_v2(
            logits, labels, penult, prototypes, config)

        # Outlier exposure loss
        out_loss = torch.tensor(0.0, device=device)
        if config.OUTLIER_LAMBDA > 0:
            outliers = generate_outliers(penult, labels, prototypes, config, device)
            out_loss = compute_outlier_loss(
                outliers, model.out, prototypes, config, device)

        total_batch_loss = loss + config.OUTLIER_LAMBDA * out_loss

        optimizer.zero_grad()
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            acc = compute_accuracy(logits, labels)

        total_loss += loss.item()
        total_ce += ce.item()
        total_proto += proto.item()
        total_rep += rep.item()
        total_out += out_loss.item()
        total_acc += acc.item()
        n_batches += 1

        # Collect features per class
        for k in range(config.NUM_KNOWN):
            mask_k = (labels == k)
            if mask_k.sum() > 0:
                all_features[k].append(penult[mask_k].detach())

    n = n_batches if n_batches > 0 else 1

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

    return (total_loss / n, total_ce / n, total_proto / n,
            total_rep / n, total_out / n, total_acc / n, new_prototypes)


@torch.no_grad()
def validate_v2(model, dataloader, config, prototypes, device="cpu"):
    """Validation epoch."""
    model.eval()
    total_loss = total_acc = 0.0
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

        loss, _, _, _ = compute_loss_v2(logits, labels, penult, prototypes, config)
        acc = compute_accuracy(logits, labels)

        total_loss += loss.item()
        total_acc += acc.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0, total_acc / n_batches if n_batches > 0 else 0


def train_v2(config, model, train_loader, val_loader, seed=42, device="cpu"):
    """
    Full training loop with contrastive loss + outlier exposure.

    Returns:
        best_model_state, training_history, final_prototypes
    """
    torch.manual_seed(seed)
    random.seed(seed)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE,
                      weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    prototypes = torch.zeros(config.NUM_KNOWN, config.FC2_UNITS)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
        "train_ce": [], "train_proto": [], "train_rep": [], "train_outlier": [],
    }

    for epoch in range(config.EPOCHS):
        train_loss, train_ce, train_proto, train_rep, train_out, train_acc, prototypes = \
            train_epoch_v2(model, train_loader, optimizer, config, prototypes, device)

        val_loss, val_acc = validate_v2(model, val_loader, config, prototypes, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["train_ce"].append(train_ce)
        history["train_proto"].append(train_proto)
        history["train_rep"].append(train_rep)
        history["train_outlier"].append(train_out)

        scheduler.step(val_loss)

        print(f"  Epoch {epoch+1:3d}/{config.EPOCHS} | "
              f"Train L={train_loss:.3f} CE={train_ce:.3f} Proto={train_proto:.4f} "
              f"Rep={train_rep:.4f} Out={train_out:.4f} Acc={train_acc:.3f} | "
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

    # Compute final prototypes
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
