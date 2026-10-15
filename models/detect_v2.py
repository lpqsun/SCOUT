#!/usr/bin/env python3
"""
Improved unknown vulnerability detection (v2).
Three improvements over v1:
  1. Per-class z-score normalization (handles Reentrancy's naturally high entropy)
  2. Continuous OOD score fusion (replaces three hard thresholds)
  3. Single learnable threshold on OOD score

No model re-training required — only modifies the decision logic.
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from tqdm import tqdm


# Reuse the existing feature extractors
from detect import mc_dropout_forward, compute_prototype_distances


def collect_per_class_stats(model, dataloader, prototypes, config, device="cpu"):
    """
    Collect per-class statistics from validation set using MC dropout.
    Only uses correctly-classified known samples.
    Returns dict: class_id -> {mean_prob, std_prob, mean_ent, std_ent, mean_dist, std_dist}
    """
    print("  Collecting per-class statistics from validation set...")

    # Gather all validation predictions
    all_entries = []
    for batch in dataloader:
        # Only process known samples
        known_mask = ~batch["is_unknown"]
        if known_mask.sum() == 0:
            continue

        mc = mc_dropout_forward(model, batch, device, T=config.MC_DROPOUT_T)
        min_dist, _ = compute_prototype_distances(mc["penult_features"], prototypes)

        for i in range(batch["labels"].size(0)):
            if not batch["is_unknown"][i].item():
                all_entries.append({
                    "true_label": batch["labels"][i].item(),
                    "pred_class": mc["pred_class"][i].item(),
                    "max_prob": mc["max_prob"][i].item(),
                    "entropy": mc["entropy"][i].item(),
                    "min_dist": min_dist[i].item(),
                })

    # Compute per-class stats from CORRECTLY classified samples only
    per_class = defaultdict(lambda: {"prob": [], "ent": [], "dist": []})
    for e in all_entries:
        if e["pred_class"] == e["true_label"]:  # only correctly classified
            k = e["true_label"]
            per_class[k]["prob"].append(e["max_prob"])
            per_class[k]["ent"].append(e["entropy"])
            per_class[k]["dist"].append(e["min_dist"])

    stats = {}
    for k in sorted(per_class.keys()):
        prob = np.array(per_class[k]["prob"])
        ent = np.array(per_class[k]["ent"])
        dist = np.array(per_class[k]["dist"])
        stats[k] = {
            "mean_prob": float(np.mean(prob)),
            "std_prob": float(np.std(prob)) + 1e-8,
            "mean_ent": float(np.mean(ent)),
            "std_ent": float(np.std(ent)) + 1e-8,
            "mean_dist": float(np.mean(dist)),
            "std_dist": float(np.std(dist)) + 1e-8,
            # Percentile-based thresholds as fallback
            "ent_p90": float(np.percentile(ent, 90)),
            "ent_p95": float(np.percentile(ent, 95)),
            "prob_p10": float(np.percentile(prob, 10)),
            "prob_p25": float(np.percentile(prob, 25)),
            "n": len(prob),
        }
        print(f"    Class {k}: n={len(prob):>4d}  "
              f"prob={stats[k]['mean_prob']:.3f}±{stats[k]['std_prob']:.3f}  "
              f"ent={stats[k]['mean_ent']:.3f}±{stats[k]['std_ent']:.3f}  "
              f"dist={stats[k]['mean_dist']:.3f}±{stats[k]['std_dist']:.3f}")

    return stats


def compute_ood_score(max_prob, entropy, min_dist, pred_class, per_class_stats,
                      anomaly_score=None, anom_stats=None):
    """
    Compute a continuous OOD score using per-class z-score normalization.
    Optionally includes anomaly head score as a 4th dimension.
    """
    if pred_class not in per_class_stats:
        return max(1.0 - max_prob, entropy / 0.6, min_dist / 0.2)

    s = per_class_stats[pred_class]

    z_prob = -(max_prob - s["mean_prob"]) / s["std_prob"]
    z_ent = (entropy - s["mean_ent"]) / s["std_ent"]
    z_dist = (min_dist - s["mean_dist"]) / s["std_dist"]

    scores = [z_prob, z_ent, z_dist]

    # Add anomaly score as 4th dimension if available
    if anomaly_score is not None and anom_stats is not None:
        z_anom = (anomaly_score - anom_stats["mean"]) / (anom_stats["std"] + 1e-8)
        scores.append(z_anom)

    return float(max(scores))


def detect_unknown_v2(model, dataloader, prototypes, config,
                       per_class_stats, ood_threshold=2.0,
                       device="cpu"):
    """
    Run improved unknown detection using per-class OOD score.

    Decision:
      ood_score <= threshold  → known  (within normal range for predicted class)
      ood_score >  threshold  → unknown (too anomalous for predicted class)

    Returns per-sample predictions with the same format as v1 detect_unknown.
    """
    all_results = []

    for batch in tqdm(dataloader, desc="  Detecting v2"):
        mc = mc_dropout_forward(model, batch, device, T=config.MC_DROPOUT_T)
        min_dist, nearest_class = compute_prototype_distances(
            mc["penult_features"], prototypes)

        B = batch["labels"].size(0)
        for i in range(B):
            mp = mc["max_prob"][i].item()
            ent = mc["entropy"][i].item()
            md = min_dist[i].item()
            pc = mc["pred_class"][i].item()

            ood = compute_ood_score(mp, ent, md, pc, per_class_stats)

            if ood <= ood_threshold:
                decision = "known"
            else:
                decision = "unknown"

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
                "ood_score": ood,
                "known_score": float(ood <= ood_threshold),
                "unknown_score": float(ood > ood_threshold),
            })

    return all_results


def calibrate_ood_threshold(model, val_loader, prototypes, config,
                             per_class_stats, device="cpu"):
    """
    Calibrate the OOD score threshold on validation set.
    Uses pseudo-unknown sampling (20% of val known samples treated as unknown).
    Searches for threshold that maximizes F1.
    """
    print("  Calibrating OOD threshold on validation set...")

    # Collect OOD scores for known validation samples
    known_scores = []
    for batch in val_loader:
        mc = mc_dropout_forward(model, batch, device, T=config.MC_DROPOUT_T)
        min_dist, _ = compute_prototype_distances(mc["penult_features"], prototypes)
        for i in range(batch["labels"].size(0)):
            if not batch["is_unknown"][i].item():
                mp = mc["max_prob"][i].item()
                ent = mc["entropy"][i].item()
                md = min_dist[i].item()
                pc = mc["pred_class"][i].item()
                ood = compute_ood_score(mp, ent, md, pc, per_class_stats)
                known_scores.append(ood)

    if not known_scores:
        return 2.0

    # Split: 80% normal, 20% pseudo-unknown
    n = len(known_scores)
    n_pseudo = max(1, int(n * config.VAL_PSEUDO_UNKNOWN_FRAC))
    indices = np.random.RandomState(42).permutation(n)
    normal_idx = indices[:n - n_pseudo]
    pseudo_idx = indices[n - n_pseudo:]

    normal_scores = [known_scores[i] for i in normal_idx]
    pseudo_scores = [known_scores[i] for i in pseudo_idx]

    # Grid search over threshold
    best_f1 = 0.0
    best_t = 2.0
    candidates = np.linspace(0.5, 5.0, 46)  # 0.5 to 5.0 in 0.1 steps

    for t in candidates:
        tp = sum(1 for s in pseudo_scores if s > t)
        fn = sum(1 for s in pseudo_scores if s <= t)
        fp = sum(1 for s in normal_scores if s > t)

        prec = tp / (tp + fp + 1e-10)
        rec = tp / (tp + fn + 1e-10)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)

        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    print(f"    Best F1={best_f1:.4f} at threshold={best_t:.2f}")
    return best_t
