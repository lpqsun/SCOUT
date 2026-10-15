#!/usr/bin/env python3
"""
Baseline methods for unknown vulnerability detection.
Section 5.3 of the paper.

Implements:
  1. Softmax Threshold  — max probability < τ → unknown
  2. OpenMax            — Weibull-based open-set recognition (Bendale & Boult, 2016)
  3. Energy-based OOD   — energy score thresholding (Liu et al., 2020)
  4. Mahalanobis Distance — class-conditional Gaussian distance

Each baseline is callable with a consistent interface:
  baseline(model, dataloader, prototypes, config, device) → list of result dicts
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect import mc_dropout_forward, compute_prototype_distances


# ==============================================================================
# Utility: collect logits & features from a dataloader
# ==============================================================================

def _build_fused(model, batch, device):
    opc = batch["opc_indices"].to(device)
    param = batch["param_indices"].to(device)
    stats = batch["stats"].to(device)
    return model.embedding(opc, param, stats, model.atom_registry)


@torch.no_grad()
def collect_logits_and_features(model, dataloader, device):
    """
    Collect logits, penultimate features, labels, and is_unknown flags
    from a full pass over the dataloader.

    Returns:
        logits: (N, K) tensor
        features: (N, D) tensor
        labels: (N,) tensor
        is_unknown: (N,) bool tensor
        categories: list of str
    """
    model.eval()
    all_logits = []
    all_features = []
    all_labels = []
    all_is_unknown = []
    all_categories = []

    for batch in dataloader:
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        mask = batch["mask"].to(device)
        fused = _build_fused(model, batch, device)
        output = model(fused, edge_index, edge_attr, mask)

        all_logits.append(output["logits"].cpu())
        all_features.append(output["penult_features"].cpu())
        all_labels.append(batch["labels"].cpu())
        all_is_unknown.append(batch["is_unknown"].cpu())
        all_categories.extend(batch.get("category", [""]))

    return (
        torch.cat(all_logits, dim=0),
        torch.cat(all_features, dim=0),
        torch.cat(all_labels, dim=0),
        torch.cat(all_is_unknown, dim=0),
        all_categories,
    )


# ==============================================================================
# Baseline 1: Softmax Threshold
# ==============================================================================

@torch.no_grad()
def baseline_softmax_threshold(model, dataloader, prototypes, config, device="cpu",
                               tau_high=None, tau_low=None):
    """
    Simple softmax max-probability thresholding (no MC Dropout, no entropy).
    Single deterministic forward pass.

    Decision:
      max_prob >= tau_high → known
      max_prob <  tau_low  → unknown
      otherwise            → ambiguous
    """
    if tau_high is None:
        tau_high = config.TAU_HIGH
    if tau_low is None:
        tau_low = config.TAU_LOW

    model.eval()
    results = []

    for batch in dataloader:
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        mask = batch["mask"].to(device)
        fused = _build_fused(model, batch, device)
        output = model(fused, edge_index, edge_attr, mask)
        probs = F.softmax(output["logits"], dim=1)
        max_prob, pred_class = probs.max(dim=1)

        B = batch["labels"].size(0)
        for i in range(B):
            mp = max_prob[i].item()
            if mp >= tau_high:
                decision = "known"
            elif mp < tau_low:
                decision = "unknown"
            else:
                decision = "ambiguous"

            results.append({
                "true_label": batch["labels"][i].item(),
                "is_unknown": batch["is_unknown"][i].item(),
                "category": batch.get("category", [""])[i] if "category" in batch else "",
                "basename": batch["basename"][i] if "basename" in batch else "",
                "max_prob": mp,
                "pred_class": pred_class[i].item(),
                "decision": decision,
            })

    return results


# ==============================================================================
# Baseline 2: OpenMax (Weibull-based Open-Set Recognition)
# ==============================================================================

def _fit_weibull_per_class(logits, labels, num_classes, tail_size=20):
    """
    Fit a Weibull distribution to the tail of distances per class.

    For each class k:
      - Compute distances of correctly-classified samples to class mean logit
      - Take the `tail_size` largest distances
      - Fit Weibull(shape, scale) using MLE approximation

    Returns:
        weibull_params: dict {class_k: {"shape": η, "scale": λ, "mean_logit": μ_k}}
    """
    try:
        from scipy.special import digamma, polygamma
    except ImportError:
        # Fallback: use numpy-based MLE approximation
        pass

    weibull_params = {}
    for k in range(num_classes):
        mask = (labels == k)
        if mask.sum() < tail_size:
            weibull_params[k] = None
            continue

        class_logits = logits[mask]  # (N_k, K)
        mean_logit = class_logits.mean(dim=0)  # (K,)

        # Distance to class mean (in logit space)
        dists = (class_logits - mean_logit.unsqueeze(0)).norm(dim=1)  # (N_k,)
        dists_sorted = dists.sort().values
        tail = dists_sorted[-tail_size:].numpy()

        # Fit Weibull using MLE approximation (method of moments)
        # Weibull PDF: f(x; η, λ) = (η/λ) * (x/λ)^(η-1) * exp(-(x/λ)^η)
        # log(x) ~ Gumbel → η ≈ 1 / std(log(x)), λ ≈ exp(mean(log(x)) + γ/η)
        log_tail = np.log(tail + 1e-10)
        shape = 1.0 / (log_tail.std() + 1e-10)
        euler_gamma = 0.5772156649
        scale = np.exp(log_tail.mean() + euler_gamma / shape)

        weibull_params[k] = {
            "shape": float(shape),
            "scale": float(scale),
            "mean_logit": mean_logit,
        }

    return weibull_params


def _openmax_recalibrate(logits, weibull_params, num_classes, alpha=3.0):
    """
    Recalibrate logits using Weibull CDF scores.
    Adds an "unknown" pseudo-class.

    Args:
        logits: (B, K)
        weibull_params: per-class Weibull parameters
        alpha: number of top classes to recalibrate

    Returns:
        openmax_probs: (B, K+1)  — last column = P(unknown)
    """
    B, K = logits.shape
    device = logits.device

    openmax_logits = torch.zeros(B, K + 1, device=device)
    openmax_logits[:, :K] = logits

    for b in range(B):
        scores, indices = logits[b].sort(descending=True)
        top_alpha = indices[:alpha]

        # Compute Weibull CDF for each top class
        w_scores = []
        for k in top_alpha.tolist():
            if weibull_params.get(k) is not None:
                wp = weibull_params[k]
                mean_logit = wp["mean_logit"].to(device)
                dist = (logits[b] - mean_logit).norm().item()
                # Weibull CDF: F(x) = 1 - exp(-(x/λ)^η)
                cdf = 1.0 - np.exp(-((dist / wp["scale"]) ** wp["shape"]))
                w_scores.append(cdf)
            else:
                w_scores.append(0.0)

        # Recalibrate: multiply by (1 - CDF)
        for j, k in enumerate(top_alpha.tolist()):
            if j < len(w_scores):
                openmax_logits[b, k] *= (1.0 - w_scores[j])

        # Unknown pseudo-logit = Σ (logit_k * CDF_k)
        unknown_score = sum(
            scores[j].item() * w_scores[j] for j in range(min(alpha, len(w_scores)))
        )
        openmax_logits[b, K] = unknown_score

    return F.softmax(openmax_logits, dim=1)


@torch.no_grad()
def baseline_openmax(model, dataloader, prototypes, config, device="cpu",
                     tail_size=20, alpha=3, tau_unknown=0.5):
    """
    OpenMax baseline for unknown vulnerability detection.

    1. Collect logits from training set
    2. Fit Weibull distributions per class
    3. On test set: recalibrate with OpenMax, threshold on P(unknown)
    """
    from detect import calibrate_thresholds

    model.eval()

    # Fit Weibull on training data
    print("  [OpenMax] Collecting training logits...")
    train_logits, _, train_labels, _, _ = collect_logits_and_features(
        model, dataloader["train"] if isinstance(dataloader, dict) else dataloader,
        device)
    weibull_params = _fit_weibull_per_class(
        train_logits, train_labels, config.NUM_KNOWN, tail_size=tail_size)

    # Predict on test set
    test_loader = dataloader["test"] if isinstance(dataloader, dict) else dataloader
    results = []
    for batch in test_loader:
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        mask = batch["mask"].to(device)
        fused = _build_fused(model, batch, device)
        output = model(fused, edge_index, edge_attr, mask)

        openmax_probs = _openmax_recalibrate(
            output["logits"].cpu(), weibull_params, config.NUM_KNOWN, alpha=alpha)

        B = batch["labels"].size(0)
        for i in range(B):
            p_unknown = openmax_probs[i, -1].item()  # last column = unknown
            known_probs = openmax_probs[i, :config.NUM_KNOWN]
            max_prob, pred_class = known_probs.max(dim=0)

            if p_unknown >= tau_unknown:
                decision = "unknown"
            else:
                decision = "known"

            results.append({
                "true_label": batch["labels"][i].item(),
                "is_unknown": batch["is_unknown"][i].item(),
                "category": batch.get("category", [""])[i] if "category" in batch else "",
                "basename": batch["basename"][i] if "basename" in batch else "",
                "max_prob": max_prob.item(),
                "p_unknown": p_unknown,
                "pred_class": pred_class.item(),
                "decision": decision,
            })

    return results


# ==============================================================================
# Baseline 3: Energy-based OOD Detection
# ==============================================================================

@torch.no_grad()
def baseline_energy_ood(model, dataloader, prototypes, config, device="cpu",
                        temperature=1.0, energy_threshold=None):
    """
    Energy-based OOD detection (Liu et al., 2020).

    Energy score: E(x) = -T * log Σ_k exp(logit_k / T)

    Higher energy → more likely OOD (unknown vulnerability).
    Threshold calibrated on validation set pseudo-unknown samples.

    If energy_threshold is None, calibrate using validation set.
    """
    model.eval()

    def compute_energy(logits, T=1.0):
        """E(x; f) = -T * logsumexp(logits / T)"""
        return -T * torch.logsumexp(logits / T, dim=1)

    # Calibrate threshold on validation set if needed
    if energy_threshold is None and isinstance(dataloader, dict) and "val" in dataloader:
        val_energies = []
        for batch in dataloader["val"]:
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device)
            mask = batch["mask"].to(device)
            fused = _build_fused(model, batch, device)
            output = model(fused, edge_index, edge_attr, mask)
            energy = compute_energy(output["logits"].cpu(), temperature)
            for i in range(batch["labels"].size(0)):
                if not batch["is_unknown"][i].item():
                    val_energies.append(energy[i].item())

        if val_energies:
            # Use 90th percentile of known class energies as threshold
            energy_threshold = float(np.percentile(val_energies, 90))
            print(f"  [Energy-OOD] Calibrated threshold = {energy_threshold:.4f}")
        else:
            energy_threshold = float('inf')

    # Predict on test set
    test_loader = dataloader["test"] if isinstance(dataloader, dict) else dataloader
    results = []
    for batch in test_loader:
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        mask = batch["mask"].to(device)
        fused = _build_fused(model, batch, device)
        output = model(fused, edge_index, edge_attr, mask)

        probs = F.softmax(output["logits"], dim=1)
        max_prob, pred_class = probs.max(dim=1)
        energy = compute_energy(output["logits"].cpu(), temperature)

        B = batch["labels"].size(0)
        for i in range(B):
            e = energy[i].item()
            if energy_threshold is not None:
                is_unknown = e > energy_threshold
                decision = "unknown" if is_unknown else "known"
            else:
                # Fallback: use default threshold
                decision = "unknown" if max_prob[i].item() < 0.5 else "known"

            results.append({
                "true_label": batch["labels"][i].item(),
                "is_unknown": batch["is_unknown"][i].item(),
                "category": batch.get("category", [""])[i] if "category" in batch else "",
                "basename": batch["basename"][i] if "basename" in batch else "",
                "max_prob": max_prob[i].item(),
                "energy": e,
                "pred_class": pred_class[i].item(),
                "decision": decision,
            })

    return results


# ==============================================================================
# Baseline 4: Mahalanobis Distance
# ==============================================================================

@torch.no_grad()
def baseline_mahalanobis(model, dataloader, prototypes, config, device="cpu",
                         epsilon=0.01, chi2_threshold=None):
    """
    Mahalanobis distance-based OOD detection.

    Fits class-conditional Gaussians in penultimate feature space (32-dim).
    Computes Mahalanobis distance to nearest class.
    Large distance → unknown.

    Uses shrinkage regularization (epsilon) for covariance estimation.
    """
    model.eval()

    # Fit class-conditional Gaussians on training data
    print("  [Mahalanobis] Fitting class-conditional Gaussians...")
    train_logits, train_feats, train_labels, _, _ = collect_logits_and_features(
        model, dataloader["train"] if isinstance(dataloader, dict) else dataloader,
        device)

    D = train_feats.size(1)  # feature dim (32)
    K = config.NUM_KNOWN
    means = torch.zeros(K, D)
    covs = torch.zeros(K, D, D)
    counts = torch.zeros(K)

    for k in range(K):
        mask = (train_labels == k)
        count = mask.sum().item()
        counts[k] = count
        if count > 1:
            class_feats = train_feats[mask]  # (N_k, D)
            means[k] = class_feats.mean(dim=0)
            # Shrinkage covariance estimate
            cov = torch.cov(class_feats.T)  # (D, D)
            covs[k] = cov + epsilon * torch.eye(D)
        elif count == 1:
            means[k] = train_feats[mask][0]
            covs[k] = torch.eye(D)

    # Invert covariances
    precisions = torch.zeros(K, D, D)
    for k in range(K):
        if counts[k] > 0:
            try:
                precisions[k] = torch.linalg.inv(covs[k])
            except torch._C._LinAlgError:
                precisions[k] = torch.eye(D) / epsilon

    # Calibrate threshold
    if chi2_threshold is None and isinstance(dataloader, dict) and "val" in dataloader:
        val_dists = []
        for batch in dataloader["val"]:
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device)
            mask = batch["mask"].to(device)
            fused = _build_fused(model, batch, device)
            output = model(fused, edge_index, edge_attr, mask)
            feats = output["penult_features"].cpu()
            for i in range(feats.size(0)):
                if not batch["is_unknown"][i].item():
                    x = feats[i]
                    md_min = float('inf')
                    for k in range(K):
                        if counts[k] > 0:
                            dx = (x - means[k]).unsqueeze(1)  # (D, 1)
                            md = (dx.T @ precisions[k] @ dx).item()
                            md_min = min(md_min, md)
                    val_dists.append(md_min)

        if val_dists:
            chi2_threshold = float(np.percentile(val_dists, 95))
            print(f"  [Mahalanobis] Calibrated threshold = {chi2_threshold:.4f}")
        else:
            chi2_threshold = float('inf')

    # Predict on test set
    test_loader = dataloader["test"] if isinstance(dataloader, dict) else dataloader
    results = []
    for batch in test_loader:
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        mask = batch["mask"].to(device)
        fused = _build_fused(model, batch, device)
        output = model(fused, edge_index, edge_attr, mask)

        probs = F.softmax(output["logits"], dim=1)
        max_prob, pred_class = probs.max(dim=1)
        feats = output["penult_features"].cpu()

        B = feats.size(0)
        for i in range(B):
            x = feats[i]
            md_min = float('inf')
            nearest_k = 0
            for k in range(K):
                if counts[k] > 0:
                    dx = (x - means[k]).unsqueeze(1)
                    md = (dx.T @ precisions[k] @ dx).item()
                    if md < md_min:
                        md_min = md
                        nearest_k = k

            if chi2_threshold is not None:
                is_unknown = md_min > chi2_threshold
            else:
                is_unknown = False

            results.append({
                "true_label": batch["labels"][i].item(),
                "is_unknown": batch["is_unknown"][i].item(),
                "category": batch.get("category", [""])[i] if "category" in batch else "",
                "basename": batch["basename"][i] if "basename" in batch else "",
                "max_prob": max_prob[i].item(),
                "mahalanobis_dist": md_min,
                "nearest_class": nearest_k,
                "pred_class": pred_class[i].item(),
                "decision": "unknown" if is_unknown else "known",
            })

    return results


# ==============================================================================
# Unified baseline runner
# ==============================================================================

BASELINE_METHODS = {
    "softmax": baseline_softmax_threshold,
    "openmax": baseline_openmax,
    "energy_ood": baseline_energy_ood,
    "mahalanobis": baseline_mahalanobis,
}

BASELINE_LABELS = {
    "softmax": "Softmax Threshold",
    "openmax": "OpenMax",
    "energy_ood": "Energy-based OOD",
    "mahalanobis": "Mahalanobis Distance",
}


def run_baseline_comparison(model, dataloaders, prototypes, config, device="cpu",
                           methods=None):
    """
    Run all baseline methods and compare with the proposed method.

    Args:
        model: trained VulnDetectionModel
        dataloaders: dict with "train", "val", "test" keys
        prototypes: known class prototypes
        config: Config
        device: compute device
        methods: list of method keys to run (default: all)

    Returns:
        dict: {method_name: {"results": [...], "metrics": {...}}}
    """
    from evaluate import compute_unknown_metrics, compute_known_metrics
    from detect import detect_unknown, calibrate_thresholds

    if methods is None:
        methods = list(BASELINE_METHODS.keys())

    print("\n" + "=" * 70)
    print("BASELINE COMPARISON (Section 5.3)")
    print("=" * 70)

    all_baselines = {}

    # ── Proposed method ──
    print("\n  Running proposed method (MC Dropout + Dual-Threshold + Prototype)...")
    tau_high, tau_low, H_low, H_high, delta_max = calibrate_thresholds(
        model, dataloaders["val"], prototypes, config, device)
    proposed_results = detect_unknown(
        model, dataloaders["test"], prototypes, config,
        tau_high=tau_high, tau_low=tau_low,
        H_low=H_low, H_high=H_high, delta_max=delta_max, device=device)
    proposed_metrics = compute_unknown_metrics(proposed_results)

    all_baselines["Proposed (Ours)"] = {
        "results": proposed_results,
        "metrics": proposed_metrics,
    }

    # ── Baselines ──
    for method_key in methods:
        label = BASELINE_LABELS.get(method_key, method_key)
        print(f"\n  Running baseline: {label}...")

        try:
            if method_key == "softmax":
                results = baseline_softmax_threshold(
                    model, dataloaders["test"], prototypes, config,
                    device=device, tau_high=tau_high, tau_low=tau_low)
            elif method_key == "openmax":
                results = baseline_openmax(
                    model, dataloaders, prototypes, config, device=device)
            elif method_key == "energy_ood":
                results = baseline_energy_ood(
                    model, dataloaders, prototypes, config, device=device)
            elif method_key == "mahalanobis":
                results = baseline_mahalanobis(
                    model, dataloaders, prototypes, config, device=device)
            else:
                continue

            metrics = compute_unknown_metrics(results)
            all_baselines[label] = {"results": results, "metrics": metrics}

        except Exception as e:
            print(f"    WARNING: {label} failed: {e}")
            all_baselines[label] = {"results": [], "metrics": {}, "error": str(e)}

    # ── Print comparison table ──
    print(f"\n{'='*70}")
    print(f"BASELINE COMPARISON RESULTS")
    print(f"{'='*70}")
    header = (f"{'Method':<30} {'UDR':>8} {'FPR':>8} {'F1@unk':>8} "
              f"{'ROC-AUC':>8} {'AUPR':>8} {'OSCA':>8}")
    print(header)
    print("-" * 78)
    for name, b in all_baselines.items():
        m = b.get("metrics", {})
        if m:
            print(f"{name:<30} {m.get('UDR',0):8.4f} {m.get('FPR',0):8.4f} "
                  f"{m.get('F1_unknown',0):8.4f} {m.get('ROC_AUC',0):8.4f} "
                  f"{m.get('AUPR',0):8.4f} {m.get('OSCA',0):8.4f}")
        else:
            print(f"{name:<30} {'FAILED':>48}")

    return all_baselines


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import Config

    cfg = Config()
    print(f"Device: {cfg.DEVICE_NAME}")
    print("Baseline methods module ready.")
    print(f"  Available baselines: {list(BASELINE_METHODS.keys())}")
