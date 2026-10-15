#!/usr/bin/env python3
"""
Unknown vulnerability detection with MC Dropout, dual-threshold, and prototype distance.
Section 4.8 of the paper.
"""

import copy
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def _build_batch_fused(model, batch, device):
    opc = batch["opc_indices"].to(device)
    param = batch["param_indices"].to(device)
    stats = batch["stats"].to(device)
    return model.embedding(opc, param, stats, model.atom_registry)


@torch.no_grad()
def mc_dropout_forward(model, batch, device, T=10):
    """
    Monte Carlo Dropout forward pass.
    Enables dropout layers, runs T stochastic forward passes.
    Returns mean probabilities and prediction entropy.
    """
    # Move tensors to device once and compute fused features once
    edge_index = batch["edge_index"].to(device)
    edge_attr = batch["edge_attr"].to(device)
    mask = batch["mask"].to(device)
    fused = _build_batch_fused(model, batch, device)

    model.train()  # enable dropout
    all_probs = []

    for _ in range(T):
        output = model(fused, edge_index, edge_attr, mask)
        probs = F.softmax(output["logits"], dim=1)
        all_probs.append(probs.cpu())

    probs = torch.stack(all_probs)  # (T, B, K)
    mean_prob = probs.mean(dim=0)   # (B, K)
    max_prob, pred_class = mean_prob.max(dim=1)  # (B,)

    # Entropy: H = -Σ p_k * log(p_k)
    entropy = -(mean_prob * (mean_prob + 1e-10).log()).sum(dim=1)  # (B,)

    # Single deterministic forward for penultimate features
    model.eval()
    output = model(fused, edge_index, edge_attr, mask)

    return {
        "mean_prob": mean_prob,
        "max_prob": max_prob,
        "pred_class": pred_class,
        "entropy": entropy,
        "penult_features": output["penult_features"].cpu(),
        "single_logits": output["logits"].cpu(),
        "attention_weights": output["attention_weights"].cpu(),
    }


def compute_prototype_distances(penult_features, prototypes):
    """
    Compute min Euclidean distance from each sample to known class prototypes.

    Args:
        penult_features: (B, D)
        prototypes: (K, D) or None
    Returns:
        min_dist: (B,)  — distance to nearest prototype
        nearest_class: (B,)  — index of nearest prototype
    """
    if prototypes is None:
        B = penult_features.size(0)
        return torch.zeros(B), torch.zeros(B, dtype=torch.long)

    # (B, D) → (B, 1, D),  (K, D) → (1, K, D)
    diff = penult_features.unsqueeze(1) - prototypes.unsqueeze(0)  # (B, K, D)
    dists = diff.pow(2).sum(dim=2).sqrt()  # (B, K)
    min_dist, nearest_class = dists.min(dim=1)
    return min_dist, nearest_class


def detect_unknown(model, dataloader, prototypes, config,
                   tau_high=None, tau_low=None, H_low=None, H_high=None,
                   delta_max=None, device="cpu"):
    """
    Run unknown vulnerability detection on a dataset.

    Returns per-sample predictions with decision labels.
    """
    if tau_high is None:
        tau_high = config.TAU_HIGH
    if tau_low is None:
        tau_low = config.TAU_LOW
    if H_low is None:
        H_low = config.ENTROPY_LOW
    if H_high is None:
        H_high = config.ENTROPY_HIGH

    all_results = []

    for batch in tqdm(dataloader, desc="  Detecting"):
        mc = mc_dropout_forward(model, batch, device, T=config.MC_DROPOUT_T)
        min_dist, nearest_class = compute_prototype_distances(
            mc["penult_features"], prototypes)

        B = batch["labels"].size(0)
        for i in range(B):
            mp = mc["max_prob"][i].item()
            ent = mc["entropy"][i].item()
            md = min_dist[i].item()
            pc = mc["pred_class"][i].item()

            # Decision rule (Section 4.8.2)
            known_cond = (mp >= tau_high and ent <= H_low)
            if delta_max is not None:
                known_cond = known_cond and (md <= delta_max)

            unknown_cond = (mp < tau_low or ent >= H_high)
            if delta_max is not None:
                unknown_cond = unknown_cond or (md > delta_max)

            if known_cond:
                decision = "known"
            elif unknown_cond:
                decision = "unknown"
            else:
                decision = "ambiguous"

            all_results.append({
                "true_label": batch["labels"][i].item(),
                "is_unknown": batch["is_unknown"][i].item(),
                "category": batch.get("category", [""])[i] if "category" in batch else "",
                "basename": batch["basename"][i] if "basename" in batch else "",
                "max_prob": mp,
                "entropy": ent,
                "min_dist": md,
                "pred_class": pc,
                "decision": decision,
                "known_score": float(known_cond),
                "unknown_score": float(unknown_cond),
            })

    return all_results


def calibrate_thresholds(model, val_loader, prototypes, config, device="cpu"):
    """
    Calibrate thresholds on validation set using pseudo-unknown sampling.
    Section 4.8.3: 20% of validation known samples are treated as "pseudo-unknown".
    """
    print("  Calibrating thresholds on validation set...")

    # Collect validation predictions
    all_known_mc = []
    for batch in val_loader:
        mc = mc_dropout_forward(model, batch, device, T=config.MC_DROPOUT_T)
        min_dist, _ = compute_prototype_distances(mc["penult_features"], prototypes)
        for i in range(batch["labels"].size(0)):
            if not batch["is_unknown"][i].item():
                all_known_mc.append({
                    "max_prob": mc["max_prob"][i].item(),
                    "entropy": mc["entropy"][i].item(),
                    "min_dist": min_dist[i].item(),
                })

    if not all_known_mc:
        return config.TAU_HIGH, config.TAU_LOW, config.ENTROPY_LOW, config.ENTROPY_HIGH, 2.0

    # Split: 80% normal, 20% pseudo-unknown
    n = len(all_known_mc)
    n_pseudo = max(1, int(n * config.VAL_PSEUDO_UNKNOWN_FRAC))
    indices = np.random.RandomState(42).permutation(n)
    normal_idx = indices[:n - n_pseudo]
    pseudo_idx = indices[n - n_pseudo:]

    normal_mc = [all_known_mc[i] for i in normal_idx]
    pseudo_mc = [all_known_mc[i] for i in pseudo_idx]

    # Set entropy thresholds
    H_low = np.percentile([m["entropy"] for m in normal_mc], 50)  # median
    H_high = np.percentile([m["entropy"] for m in pseudo_mc], 75)

    # Set distance threshold
    delta_max = np.percentile([m["min_dist"] for m in normal_mc], 95)

    # Grid search over tau values
    best_f1 = 0.0
    best_tau_high = config.TAU_HIGH
    best_tau_low = config.TAU_LOW
    best_params = (best_tau_high, best_tau_low, H_low, H_high, float(delta_max))

    for tau_h in [0.75, 0.80, 0.85, 0.90]:
        for tau_l in [0.35, 0.40, 0.45, 0.50]:
            tp = 0
            fp = 0
            fn = 0
            for m in pseudo_mc:
                known = (m["max_prob"] >= tau_h and m["entropy"] <= H_low and m["min_dist"] <= delta_max)
                if not known:
                    tp += 1
                else:
                    fn += 1
            for m in normal_mc:
                known = (m["max_prob"] >= tau_h and m["entropy"] <= H_low and m["min_dist"] <= delta_max)
                if not known:
                    fp += 1

            prec = tp / (tp + fp + 1e-10)
            rec = tp / (tp + fn + 1e-10)
            f1 = 2 * prec * rec / (prec + rec + 1e-10)
            if f1 > best_f1:
                best_f1 = f1
                best_tau_high = tau_h
                best_tau_low = tau_l
                best_params = (tau_h, tau_l, float(H_low), float(H_high), float(delta_max))

    print(f"    Best F1={best_f1:.4f}, tau_high={best_tau_high}, tau_low={best_tau_low}, "
          f"H_low={H_low:.3f}, H_high={H_high:.3f}, dist={delta_max:.3f}")
    return best_params
