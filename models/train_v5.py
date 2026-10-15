#!/usr/bin/env python3
"""
V5 Training: Joint optimization with normal traces (Route 1) + enhanced outlier exposure (Route 2).

Route 1: Normal traces as real negative samples
  - Entropy maximization: force normal trace predictions toward uniform distribution
  - Prototype repulsion: push normal features away from ALL known vulnerability prototypes
  - Normal features are used to generate better synthetic outliers

Route 2: Enhanced outlier exposure
  - Inter-class interpolation (from V2)
  - Normal-trace-based perturbation: use real normal features as outlier bases
  - Adaptive noise scaling based on feature-space density

Training loss:
  L = L_CE(known_vuln) + λ_proto * L_compactness + λ_rep * L_repulsion
    + λ_out * L_outlier(synthetic)
    + λ_normal * [L_entropy(normal) + L_proto_rep(normal)]
"""

import os, sys, json, copy, random, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# V5 Loss Functions
# ============================================================================
def compute_loss_v5(logits, labels, penult_features, class_prototypes, config):
    """
    Loss for known vulnerability samples.
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

    # 2. Prototype Compactness
    proto_loss = torch.tensor(0.0, device=device)
    if class_prototypes is not None and config.PROTOTYPE_LAMBDA > 0:
        for k in range(K):
            mask = (labels == k)
            if mask.sum() > 0:
                diff = penult_features[mask] - class_prototypes[k].unsqueeze(0).to(device)
                proto_loss += (diff.pow(2)).sum()
        proto_loss = proto_loss / B

    # 3. Inter-class Repulsion
    rep_loss = torch.tensor(0.0, device=device)
    if class_prototypes is not None and config.REPULSION_LAMBDA > 0:
        proto_cpu = class_prototypes.to(device)
        n_pairs = 0
        for i in range(K):
            for j in range(i + 1, K):
                dist_ij = (proto_cpu[i] - proto_cpu[j]).pow(2).sum().sqrt()
                rep_loss += F.relu(config.REPULSION_MARGIN - dist_ij)
                n_pairs += 1
        if n_pairs > 0:
            rep_loss = rep_loss / n_pairs

    return ce_loss + config.PROTOTYPE_LAMBDA * proto_loss + config.REPULSION_LAMBDA * rep_loss, ce_loss, proto_loss, rep_loss


def compute_normal_loss(penult_features, model_out_layer, class_prototypes, config, device):
    """
    Route 1: Loss for normal (non-vulnerable) traces.
    Forces the model to:
      1. Output uniform softmax (entropy maximization)
      2. Stay far from ALL vulnerability prototypes (repulsion)
    """
    K = config.NUM_KNOWN
    B = penult_features.size(0)
    if B == 0:
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

    # 1. Entropy maximization: push toward uniform distribution
    logits = model_out_layer(penult_features)
    probs = F.softmax(logits, dim=1)
    target = torch.ones(B, K, device=device) / K
    entropy_loss = F.kl_div(probs.log(), target, reduction='batchmean')

    # 2. Prototype repulsion: push away from nearest prototype
    proto = class_prototypes.to(device)
    dists = torch.cdist(penult_features, proto)  # (B, K)
    min_dist, nearest = dists.min(dim=1)
    # Penalize if closer than margin to ANY prototype
    proto_rep_loss = F.relu(config.NORMAL_PROTO_MARGIN - min_dist).mean()

    return entropy_loss + config.NORMAL_PROTO_LAMBDA * proto_rep_loss, entropy_loss, proto_rep_loss


def generate_outliers_v5(penult_features, labels, normal_features, prototypes, config, device):
    """
    Route 2: Enhanced outlier generation using normal trace features.

    Three strategies:
      1. Inter-class prototype interpolation (from V2)
      2. Normal-trace-based perturbation (NEW): use real normal features as base
      3. Adaptive feature perturbation (from V2, enhanced)
    """
    B, D = penult_features.shape
    K = prototypes.shape[0]
    n = config.OUTLIER_COUNT
    outliers = []
    proto_cpu = prototypes.to(device)

    third = max(1, n // 3)

    # Strategy 1: Inter-class interpolation (V2)
    for _ in range(third):
        i, j = random.sample(range(K), 2)
        alpha = random.random()
        interp = alpha * proto_cpu[i] + (1 - alpha) * proto_cpu[j]
        noise = config.OUTLIER_NOISE * torch.randn(D, device=device)
        outliers.append(interp + noise)

    # Strategy 2 (NEW): Normal-trace-based outliers
    # Use real normal features as "realistic" outlier bases
    if normal_features is not None and normal_features.size(0) > 0:
        for _ in range(third):
            idx = random.randint(0, normal_features.size(0) - 1)
            nf = normal_features[idx].detach().to(device)
            # Perturb the normal feature moderately
            noise = config.OUTLIER_NOISE * 0.7 * torch.randn(D, device=device)
            outliers.append(nf + noise)
    else:
        # Fallback: use inter-class interpolation
        for _ in range(third):
            i, j = random.sample(range(K), 2)
            alpha = random.random()
            interp = alpha * proto_cpu[i] + (1 - alpha) * proto_cpu[j]
            noise = config.OUTLIER_NOISE * 1.5 * torch.randn(D, device=device)
            outliers.append(interp + noise)

    # Strategy 3: Real feature perturbation (V2, enhanced)
    for _ in range(n - 2 * third):
        idx = random.randint(0, B - 1)
        feat = penult_features[idx].detach()
        # Adaptive noise: scale by distance to nearest prototype
        with torch.no_grad():
            dist = torch.cdist(feat.unsqueeze(0), proto_cpu).min().item()
        noise_scale = config.OUTLIER_NOISE * (1.0 + 0.5 * dist)
        noise = noise_scale * torch.randn(D, device=device)
        outliers.append(feat + noise)

    return torch.stack(outliers)


def compute_outlier_loss_v5(outlier_features, model_out_layer, class_prototypes, config, device):
    """Same as V2: entropy max + prototype repulsion for outliers."""
    K = config.NUM_KNOWN
    n_out = outlier_features.size(0)

    logits = model_out_layer(outlier_features)
    probs = F.softmax(logits, dim=1)
    target = torch.ones(n_out, K, device=device) / K
    entropy_loss = F.kl_div(probs.log(), target, reduction='batchmean')

    proto = class_prototypes.to(device)
    dists = torch.cdist(outlier_features, proto)
    min_dist, _ = dists.min(dim=1)
    proto_loss = F.relu(config.OUTLIER_PROTO_MARGIN - min_dist).mean()

    return entropy_loss + config.OUTLIER_PROTO_LAMBDA * proto_loss


# ============================================================================
# V5 Training Epoch
# ============================================================================
def _build_batch_fused(model, batch, device):
    opc = batch["opc_indices"].to(device)
    param = batch["param_indices"].to(device)
    stats = batch["stats"].to(device)
    return model.embedding(opc, param, stats, model.atom_registry)


def train_epoch_v5(model, dataloader, normal_loader, optimizer, config, prototypes,
                    device="cpu", normal_features_cache=None):
    """V5 training epoch with normal traces and enhanced outliers."""
    model.train()
    total_loss = total_ce = total_proto = total_rep = total_out = total_norm = 0.0
    total_acc = 0.0
    n_batches = 0

    all_features = [[] for _ in range(config.NUM_KNOWN)]
    collected_normal_features = []

    # Create normal data iterator
    normal_iter = iter(normal_loader) if normal_loader is not None else None

    for batch in dataloader:
        labels = batch["labels"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        mask = batch["mask"].to(device)
        fused = _build_batch_fused(model, batch, device)

        output = model(fused, edge_index, edge_attr, mask)
        logits = output["logits"]
        penult = output["penult_features"]

        # === Core loss for known vulnerability samples ===
        loss, ce, proto, rep = compute_loss_v5(
            logits, labels, penult, prototypes, config)

        # === Route 2: Enhanced outlier exposure ===
        out_loss = torch.tensor(0.0, device=device)
        if config.OUTLIER_LAMBDA > 0:
            # Get normal features from cache or current batch
            nf = normal_features_cache
            if nf is not None and nf.size(0) > 0:
                # Random subset for efficiency
                idx = torch.randint(0, nf.size(0), (min(32, nf.size(0)),))
                nf_batch = nf[idx].to(device)
            else:
                nf_batch = None

            outliers = generate_outliers_v5(
                penult, labels, nf_batch, prototypes, config, device)
            out_loss = compute_outlier_loss_v5(
                outliers, model.out, prototypes, config, device)

        total_batch_loss = loss + config.OUTLIER_LAMBDA * out_loss

        # === Route 1 (NEW): Normal trace loss ===
        norm_loss = torch.tensor(0.0, device=device)
        if config.NORMAL_LAMBDA > 0 and normal_iter is not None:
            try:
                normal_batch = next(normal_iter)
            except StopIteration:
                normal_iter = iter(normal_loader)
                normal_batch = next(normal_iter)

            n_fused = _build_batch_fused(model, normal_batch, device)
            n_output = model(n_fused, normal_batch["edge_index"].to(device),
                            normal_batch["edge_attr"].to(device),
                            normal_batch["mask"].to(device))
            n_penult = n_output["penult_features"]

            norm_loss, _, _ = compute_normal_loss(
                n_penult, model.out, prototypes, config, device)
            collected_normal_features.append(n_penult.detach().cpu())

        total_batch_loss = total_batch_loss + config.NORMAL_LAMBDA * norm_loss

        optimizer.zero_grad()
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            acc = (logits.argmax(dim=1) == labels).float().mean()

        total_loss += total_batch_loss.item()
        total_ce += ce.item()
        total_proto += proto.item()
        total_rep += rep.item()
        total_out += out_loss.item()
        total_norm += norm_loss.item()
        total_acc += acc.item()
        n_batches += 1

        # Collect features per class
        for k in range(config.NUM_KNOWN):
            mask_k = (labels == k)
            if mask_k.sum() > 0:
                all_features[k].append(penult[mask_k].detach())

    n = max(n_batches, 1)

    # Update prototypes via EMA
    new_prototypes = prototypes.clone()
    for k in range(config.NUM_KNOWN):
        if all_features[k]:
            cat_feats = torch.cat(all_features[k], dim=0)
            class_mean = cat_feats.mean(dim=0).cpu()
            if prototypes[k].sum() != 0:
                new_prototypes[k] = (config.PROTOTYPE_EMA_ALPHA * prototypes[k] +
                                     (1 - config.PROTOTYPE_EMA_ALPHA) * class_mean)
            else:
                new_prototypes[k] = class_mean

    # Build normal features cache for next epoch
    if collected_normal_features:
        normal_cache = torch.cat(collected_normal_features, dim=0)
    else:
        normal_cache = normal_features_cache

    return (total_loss/n, total_ce/n, total_proto/n, total_rep/n,
            total_out/n, total_norm/n, total_acc/n, new_prototypes, normal_cache)


@torch.no_grad()
def validate_v5(model, dataloader, config, prototypes, device="cpu"):
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
        loss, _, _, _ = compute_loss_v5(logits, labels, penult, prototypes, config)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        total_loss += loss.item()
        total_acc += acc.item()
        n_batches += 1
    return total_loss/max(n_batches,1), total_acc/max(n_batches,1)


def train_v5(config, model, train_loader, val_loader, normal_loader, seed=42, device="cpu",
             early_stop=True):
    """V5 training with normal traces + enhanced outlier exposure."""
    torch.manual_seed(seed)
    random.seed(seed)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE,
                      weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    prototypes = torch.zeros(config.NUM_KNOWN, config.FC2_UNITS)
    normal_cache = None

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "train_ce": [], "train_outlier": [], "train_normal": []}

    for epoch in range(config.EPOCHS):
        (train_loss, train_ce, train_proto, train_rep, train_out,
         train_norm, train_acc, prototypes, normal_cache) = train_epoch_v5(
            model, train_loader, normal_loader, optimizer, config, prototypes,
            device, normal_cache)

        val_loss, val_acc = validate_v5(model, val_loader, config, prototypes, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["train_ce"].append(train_ce)
        history["train_outlier"].append(train_out)
        history["train_normal"].append(train_norm)

        scheduler.step(val_loss)

        print(f"  Epoch {epoch+1:3d}/{config.EPOCHS} | "
              f"L={train_loss:.3f} CE={train_ce:.3f} Out={train_out:.4f} "
              f"Norm={train_norm:.4f} Acc={train_acc:.3f} | "
              f"Val L={val_loss:.3f} Acc={val_acc:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if early_stop and patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Keep best-val-loss state (early stopping) or the final epoch's state.
    if early_stop:
        model.load_state_dict(best_state)
        keep_state = best_state
    else:
        keep_state = copy.deepcopy(model.state_dict())  # final epoch weights

    # Final prototypes
    print("  Computing final prototypes...")
    model.eval()
    final_prototypes = torch.zeros(config.NUM_KNOWN, config.FC2_UNITS)
    proto_counts = torch.zeros(config.NUM_KNOWN)
    with torch.no_grad():
        for batch in train_loader:
            labels = batch["labels"]
            fused = _build_batch_fused(model, batch, device)
            output = model(fused, batch["edge_index"].to(device),
                          batch["edge_attr"].to(device), batch["mask"].to(device))
            penult = output["penult_features"].cpu()
            for k in range(config.NUM_KNOWN):
                mask_k = (labels == k)
                n_k = mask_k.sum().item()
                if n_k > 0:
                    final_prototypes[k] += penult[mask_k].sum(dim=0)
                    proto_counts[k] += n_k
    for k in range(config.NUM_KNOWN):
        if proto_counts[k] > 0:
            final_prototypes[k] /= proto_counts[k]

    return keep_state, history, final_prototypes


# ============================================================================
# Normal Trace Dataset
# ============================================================================
class NormalTraceDataset(torch.utils.data.Dataset):
    """Loads normal traces using the same data pipeline as VulnTraceDataset."""

    def __init__(self, config, opc_vocab, param_vocab, atom_registry):
        self.config = config
        self.opc_vocab = opc_vocab
        self.param_vocab = param_vocab
        self.atom_registry = atom_registry
        self.L_max = config.L_MAX

        normal_dir = os.path.join(config.DATA_ROOT, "normal_traces_ParamFeatures")
        param_files = sorted([f for f in os.listdir(normal_dir)
                             if f.endswith("_param_only.txt")])

        self.samples = []
        for pf in param_files:
            basename = pf.replace("_param_only.txt", "")
            self.samples.append({
                "basename": basename,
                "param_dir": normal_dir,
                "trace_dir": os.path.join(config.DATA_ROOT, "normal_traces"),
            })

        print(f"  [normal] {len(self.samples)} samples loaded from {normal_dir}")

    def __len__(self):
        return len(self.samples)

    def _load_param_seq(self, sample):
        path = os.path.join(sample["param_dir"], sample["basename"] + "_param_only.txt")
        tokens = []
        with open(path, 'r') as f:
            for line in f:
                tok = line.strip()
                if tok:
                    tokens.append(tok)
        return tokens

    def _load_opcode_seq(self, sample):
        path = os.path.join(sample["param_dir"], sample["basename"] + "_param_features.json")
        with open(path, 'r') as f:
            data = json.load(f)
        return [s["opcode"] for s in data.get("steps", [])]

    def _load_dep_edges(self, sample, max_len):
        path = os.path.join(sample["param_dir"], sample["basename"] + "_dep_matrix.json")
        with open(path, 'r') as f:
            dep_dict = json.load(f)
        edges = []
        for src_str, dsts in dep_dict.items():
            src = int(src_str)
            if src >= max_len:
                continue
            for dst in dsts[:10]:
                if dst < max_len:
                    edges.append([src, dst])
        if not edges:
            return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.float32)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.ones(edge_index.shape[1], dtype=torch.float32)
        return edge_index, edge_attr

    def _load_stats(self, sample):
        path = os.path.join(sample["param_dir"], sample["basename"] + "_stats.npy")
        if os.path.exists(path):
            return np.load(path).astype(np.float32)
        return np.zeros((1, 4), dtype=np.float32)

    def _pad_or_truncate_1d(self, seq, max_len, pad_val=0):
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq = seq + [pad_val] * (max_len - len(seq))
        return torch.tensor(seq, dtype=torch.long)

    def _pad_or_truncate_2d(self, arr, max_len):
        L, D = arr.shape
        if L > max_len:
            arr = arr[:max_len, :]
        elif L < max_len:
            pad = np.zeros((max_len - L, D), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        return torch.from_numpy(arr)

    @staticmethod
    def collate_fn(batch):
        """Same collate logic as VulnTraceDataset."""
        batch_max_len = max(item["length"] for item in batch)
        B = len(batch)
        param_indices = torch.stack([item["param_indices"][:batch_max_len] for item in batch])
        opc_indices = torch.stack([item["opc_indices"][:batch_max_len] for item in batch])
        stats = torch.stack([item["stats"][:batch_max_len, :] for item in batch])
        mask = torch.stack([item["mask"][:batch_max_len] for item in batch])
        labels = torch.stack([item["label"] for item in batch])
        edge_indices_list, edge_attrs_list = [], []
        for i, item in enumerate(batch):
            ei = item["edge_index"].clone()
            if ei.numel() > 0:
                ei[0, :] += i * batch_max_len
                ei[1, :] += i * batch_max_len
            edge_indices_list.append(ei)
            edge_attrs_list.append(item["edge_attr"])
        edge_index = torch.cat(edge_indices_list, dim=1) if edge_indices_list else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.cat(edge_attrs_list, dim=0) if edge_attrs_list else torch.zeros((0,), dtype=torch.float32)
        basenames = [item["basename"] for item in batch]
        categories = [item["category"] for item in batch]
        lengths = [item["length"] for item in batch]
        return {
            "param_indices": param_indices, "opc_indices": opc_indices,
            "stats": stats, "mask": mask, "edge_index": edge_index,
            "edge_attr": edge_attr, "labels": labels,
            "is_unknown": torch.zeros(B, dtype=torch.bool),
            "basename": basenames, "category": categories, "length": lengths,
        }

    def __getitem__(self, idx):
        sample = self.samples[idx]
        param_tokens = self._load_param_seq(sample)
        opcode_tokens = self._load_opcode_seq(sample)
        stats = self._load_stats(sample)
        orig_len = len(param_tokens)
        L = min(orig_len, self.L_max)

        param_indices = self.param_vocab.encode(param_tokens)
        param_indices = self._pad_or_truncate_1d(param_indices, self.L_max)
        opc_indices = self.opc_vocab.encode(opcode_tokens)
        opc_indices = self._pad_or_truncate_1d(opc_indices, self.L_max)
        stats_tensor = self._pad_or_truncate_2d(stats, self.L_max)
        edge_index, edge_attr = self._load_dep_edges(sample, L)
        mask = torch.zeros(self.L_max, dtype=torch.bool)
        mask[:L] = True

        return {
            "param_indices": param_indices, "opc_indices": opc_indices,
            "stats": stats_tensor, "edge_index": edge_index,
            "edge_attr": edge_attr, "mask": mask, "length": L,
            "label": torch.tensor(-1, dtype=torch.long),
            "is_unknown": False, "basename": sample["basename"],
            "category": "normal",
        }


if __name__ == "__main__":
    print("train_v5.py — ready for import.")
    print("Use run_experiments_v5.py to run full training.")
