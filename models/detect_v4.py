#!/usr/bin/env python3
"""
V4 Unknown Vulnerability Detection: Multi-Signal Enhancement.

Adds 3 new signals on top of V3 (max_prob, entropy, min_dist, anomaly_score):
  1. Energy Score       — logsumexp over all logits (Liu et al., 2020)
  2. d₁/d₂ Ratio        — nearest / second-nearest prototype distance (ambiguity)
  3. Mahalanobis Dist   — per-class Gaussian distance (Lee et al., 2018)

Plus improved fusion strategies (l2 norm, sum_positive) replacing max().
Zero training cost — all signals are post-hoc.
"""

import os, sys, json, pickle, argparse
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from detect import mc_dropout_forward, compute_prototype_distances
from evaluate import compute_unknown_metrics


# ============================================================================
# New Signal 1: Energy Score
# ============================================================================
def compute_energy_score(logits, T=1.0):
    """
    Energy-based score: E(x;f) = -T · log Σ exp(f_k/T)
    Lower (more negative) energy = more in-distribution.
    Uses MC-dropout single-forward logits.
    """
    return -T * torch.logsumexp(logits / T, dim=1)


# ============================================================================
# New Signal 2: Multi-Prototype Distances
# ============================================================================
def compute_multi_proto_distances(penult_features, prototypes):
    """
    Extended prototype distances.
    Returns:
        d1:        distance to nearest prototype
        d2:        distance to second-nearest prototype
        ratio:     d1 / d2  (0~1, high = ambiguous between two classes)
        mean_dist: average distance to all prototypes
    """
    diff = penult_features.unsqueeze(1) - prototypes.unsqueeze(0)  # (B, K, D)
    dists = diff.pow(2).sum(dim=2).sqrt()  # (B, K)

    sorted_dists, _ = dists.sort(dim=1)  # (B, K) ascending
    d1 = sorted_dists[:, 0]
    d2 = sorted_dists[:, 1] if sorted_dists.size(1) > 1 else d1 + 1.0
    ratio = d1 / (d2 + 1e-8)        # 0 = clearly one class, ~1 = ambiguous
    mean_dist = dists.mean(dim=1)

    return d1, d2, ratio, mean_dist


# ============================================================================
# New Signal 3: Mahalanobis Distance
# ============================================================================
def fit_mahalanobis_stats(model, train_loader, config, device):
    """
    Fit per-class mean and inverse covariance on training penultimate features.
    One pass through training set (~1-2 min on GPU).
    """
    print("  Fitting Mahalanobis per-class Gaussians...")
    per_class_feats = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="  Collecting train features"):
            fused = model.embedding(
                batch["opc_indices"].to(device),
                batch["param_indices"].to(device),
                batch["stats"].to(device),
                model.atom_registry,
            )
            output = model(fused, batch["edge_index"].to(device),
                           batch["edge_attr"].to(device), batch["mask"].to(device))
            feats = output["penult_features"].cpu()
            labels = batch["labels"]
            for i in range(len(labels)):
                per_class_feats[labels[i].item()].append(feats[i])

    reg_strength = 0.05  # regularization for stable inversion
    stats = {}
    for k in sorted(per_class_feats.keys()):
        feats = torch.stack(per_class_feats[k])  # (N_k, D)
        mean = feats.mean(dim=0)                 # (D,)
        centered = feats - mean.unsqueeze(0)
        cov = (centered.T @ centered) / (len(feats) - 1)
        # Tikhonov regularization
        reg = reg_strength * torch.eye(cov.shape[0])
        try:
            inv_cov = torch.inverse(cov + reg)
            stats[k] = {"mean": mean, "inv_cov": inv_cov, "n": len(feats)}
            print(f"    Class {k}: n={len(feats)}, cov cond={torch.linalg.cond(cov):.1f}")
        except RuntimeError:
            # Fallback: diagonal approximation
            diag = torch.diag(cov) + reg_strength
            inv_cov = torch.diag(1.0 / diag)
            stats[k] = {"mean": mean, "inv_cov": inv_cov, "n": len(feats)}
            print(f"    Class {k}: n={len(feats)} (diagonal fallback)")

    return stats


def compute_mahalanobis_distance(penult_features, pred_class, mahal_stats):
    """Compute Mahalanobis distance to predicted class center."""
    B = penult_features.size(0)
    dists = torch.zeros(B)
    for i in range(B):
        pc = pred_class[i].item()
        if pc in mahal_stats:
            s = mahal_stats[pc]
            diff = penult_features[i] - s["mean"]
            d2 = diff.unsqueeze(0) @ s["inv_cov"] @ diff.unsqueeze(1)
            dists[i] = torch.sqrt(torch.clamp(d2.squeeze(), min=0))
        else:
            dists[i] = 0.0
    return dists


# ============================================================================
# Extended Per-Class Statistics (7 signals: 3 base + 3 new + anomaly)
# ============================================================================
SIGNAL_NAMES = ["prob", "ent", "dist", "anom", "energy", "d1d2_ratio", "mean_proto_dist", "mahal"]

def collect_per_class_stats_v4(model, anomaly_head, dataloader, prototypes,
                                mahal_stats, config, device):
    """
    Collect per-class mean/std for all signals from correctly-classified known val samples.
    """
    print("  Collecting per-class stats (V4: up to 8 signals)...")

    per_class = defaultdict(lambda: {name: [] for name in SIGNAL_NAMES})

    for batch in tqdm(dataloader, desc="  V4 stats"):
        known_mask = ~batch["is_unknown"]
        if known_mask.sum() == 0:
            continue

        mc = mc_dropout_forward(model, batch, device, T=config.MC_DROPOUT_T)
        d1, d2, ratio, mean_dist = compute_multi_proto_distances(
            mc["penult_features"], prototypes)
        energy = compute_energy_score(mc["single_logits"])
        mahal = compute_mahalanobis_distance(
            mc["penult_features"], mc["pred_class"], mahal_stats)

        # Anomaly scores
        with torch.no_grad():
            anom = anomaly_head(mc["penult_features"].to(device)).cpu()

        for i in range(batch["labels"].size(0)):
            if not batch["is_unknown"][i].item():
                pred = mc["pred_class"][i].item()
                true = batch["labels"][i].item()
                if pred == true:
                    per_class[true]["prob"].append(mc["max_prob"][i].item())
                    per_class[true]["ent"].append(mc["entropy"][i].item())
                    per_class[true]["dist"].append(d1[i].item())
                    per_class[true]["anom"].append(anom[i].item())
                    per_class[true]["energy"].append(energy[i].item())
                    per_class[true]["d1d2_ratio"].append(ratio[i].item())
                    per_class[true]["mean_proto_dist"].append(mean_dist[i].item())
                    per_class[true]["mahal"].append(mahal[i].item())

    stats = {}
    for k in sorted(per_class.keys()):
        s = {}
        for name in SIGNAL_NAMES:
            arr = np.array(per_class[k][name])
            s[f"mean_{name}"] = float(np.mean(arr))
            s[f"std_{name}"] = float(np.std(arr)) + 1e-8
        s["n"] = len(per_class[k]["prob"])
        stats[k] = s
        print(f"    Class {k}: n={s['n']:>4d}  "
              f"energy={s['mean_energy']:.2f}±{s['std_energy']:.2f}  "
              f"d1/d2={s['mean_d1d2_ratio']:.3f}  mahal={s['mean_mahal']:.2f}")

    return stats


# ============================================================================
# V4 OOD Score Computation
# ============================================================================
def compute_ood_score_v4(max_prob, entropy, min_dist, pred_class, per_class_stats,
                          anomaly_score=None, energy=None, d1d2_ratio=None,
                          mean_proto_dist=None, mahal=None, fusion="l2"):
    """
    V4 OOD score with 7-8 signals and flexible fusion.

    Args:
        fusion: "max" | "l2" | "sum_positive"

    Each signal is z-normalized per-class, then fused.
    """
    s = per_class_stats.get(pred_class)
    if s is None:
        return 5.0

    z_all = []

    # --- Base 3 signals (always active) ---
    z_all.append(-(max_prob - s["mean_prob"]) / s["std_prob"])   # low prob → OOD
    z_all.append((entropy - s["mean_ent"]) / s["std_ent"])        # high ent → OOD
    z_all.append((min_dist - s["mean_dist"]) / s["std_dist"])     # far from proto → OOD

    # --- Anomaly head (from V3) ---
    if anomaly_score is not None and "mean_anom" in s:
        z_all.append((anomaly_score - s["mean_anom"]) / s["std_anom"])

    # --- Energy (more negative = in-dist; negate so positive = OOD) ---
    if energy is not None and "mean_energy" in s:
        z_all.append(-(energy - s["mean_energy"]) / s["std_energy"])

    # --- d1/d2 ratio (high = ambiguous = OOD) ---
    if d1d2_ratio is not None and "mean_d1d2_ratio" in s:
        z_all.append((d1d2_ratio - s["mean_d1d2_ratio"]) / s["std_d1d2_ratio"])

    # --- Mean prototype distance ---
    if mean_proto_dist is not None and "mean_mean_proto_dist" in s:
        z_all.append((mean_proto_dist - s["mean_mean_proto_dist"]) / s["std_mean_proto_dist"])

    # --- Mahalanobis distance ---
    if mahal is not None and "mean_mahal" in s:
        z_all.append((mahal - s["mean_mahal"]) / s["std_mahal"])

    z = np.array(z_all)

    if fusion == "max":
        return float(np.max(z))
    elif fusion == "l2":
        return float(np.sqrt(np.sum(z ** 2)))
    elif fusion == "sum_positive":
        return float(np.sum(np.clip(z, 0, None)))
    else:
        return float(np.max(z))


# ============================================================================
# V4 Detection
# ============================================================================
def detect_v4(model, anomaly_head, dataloader, prototypes, mahal_stats,
               per_class_stats, ood_threshold, config, device="cpu",
               fusion="l2", signals="all"):
    """
    Run V4 unknown detection with configurable signals and fusion.

    Args:
        signals: "base3" | "v3" (4 signals) | "all" (7-8 signals)
        fusion: "max" | "l2" | "sum_positive"
    """
    anomaly_head.eval()
    all_results = []

    for batch in tqdm(dataloader, desc=f"  Detecting V4 ({fusion})"):
        mc = mc_dropout_forward(model, batch, device, T=config.MC_DROPOUT_T)
        d1, d2, ratio, mean_dist = compute_multi_proto_distances(
            mc["penult_features"], prototypes)
        energy = compute_energy_score(mc["single_logits"])
        mahal = compute_mahalanobis_distance(
            mc["penult_features"], mc["pred_class"], mahal_stats)

        with torch.no_grad():
            anom = anomaly_head(mc["penult_features"].to(device)).cpu()

        B = batch["labels"].size(0)
        for i in range(B):
            mp = mc["max_prob"][i].item()
            ent = mc["entropy"][i].item()
            md = d1[i].item()
            pc = mc["pred_class"][i].item()
            es = energy[i].item()
            rr = ratio[i].item()
            mpd = mean_dist[i].item()
            mh = mahal[i].item()
            an = anom[i].item()

            # Select which signals to use
            kwargs = {"max_prob": mp, "entropy": ent, "min_dist": md,
                       "pred_class": pc, "per_class_stats": per_class_stats}

            if signals in ("v3", "all"):
                kwargs["anomaly_score"] = an

            if signals == "all":
                kwargs["energy"] = es
                kwargs["d1d2_ratio"] = rr
                kwargs["mean_proto_dist"] = mpd
                kwargs["mahal"] = mh

            ood = compute_ood_score_v4(**kwargs, fusion=fusion)

            decision = "unknown" if ood > ood_threshold else "known"

            all_results.append({
                "true_label": batch["labels"][i].item(),
                "is_unknown": batch["is_unknown"][i].item(),
                "category": batch.get("category", [""])[i] if "category" in batch else "",
                "basename": batch["basename"][i] if "basename" in batch else "",
                "max_prob": mp, "entropy": ent, "min_dist": md,
                "pred_class": pc, "decision": decision,
                "ood_score": ood, "anomaly_score": an,
                "energy": es, "d1d2_ratio": rr,
                "mean_proto_dist": mpd, "mahal": mh,
            })

    return all_results


# ============================================================================
# OOD Threshold Calibration (V4)
# ============================================================================
def calibrate_ood_threshold_v4(model, anomaly_head, val_loader, prototypes,
                                mahal_stats, per_class_stats, config, device="cpu",
                                fusion="l2", signals="all"):
    """
    Calibrate OOD threshold on validation set (same method as V2, adapted for V4).
    """
    print(f"  Calibrating OOD threshold (V4, {fusion}, {signals})...")

    known_scores = []
    for batch in val_loader:
        mc = mc_dropout_forward(model, batch, device, T=config.MC_DROPOUT_T)
        d1, d2, ratio, mean_dist = compute_multi_proto_distances(
            mc["penult_features"], prototypes)
        energy = compute_energy_score(mc["single_logits"])
        mahal = compute_mahalanobis_distance(
            mc["penult_features"], mc["pred_class"], mahal_stats)
        with torch.no_grad():
            anom = anomaly_head(mc["penult_features"].to(device)).cpu()

        for i in range(batch["labels"].size(0)):
            if not batch["is_unknown"][i].item():
                kwargs = {
                    "max_prob": mc["max_prob"][i].item(),
                    "entropy": mc["entropy"][i].item(),
                    "min_dist": d1[i].item(),
                    "pred_class": mc["pred_class"][i].item(),
                    "per_class_stats": per_class_stats,
                }
                if signals in ("v3", "all"):
                    kwargs["anomaly_score"] = anom[i].item()
                if signals == "all":
                    kwargs.update({
                        "energy": energy[i].item(),
                        "d1d2_ratio": ratio[i].item(),
                        "mean_proto_dist": mean_dist[i].item(),
                        "mahal": mahal[i].item(),
                    })
                ood = compute_ood_score_v4(**kwargs, fusion=fusion)
                known_scores.append(ood)

    if not known_scores:
        return 2.0

    n = len(known_scores)
    n_pseudo = max(1, int(n * config.VAL_PSEUDO_UNKNOWN_FRAC))
    indices = np.random.RandomState(42).permutation(n)
    normal_scores = [known_scores[i] for i in indices[:n - n_pseudo]]
    pseudo_scores = [known_scores[i] for i in indices[n - n_pseudo:]]

    best_f1 = 0.0
    best_t = 2.0
    candidates = np.linspace(0.5, 10.0, 96)  # wider range for L2 fusion

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


# ============================================================================
# Main: Compare V3 vs V4 across fusion strategies
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="output/seed_42_v2/checkpoint_best.pt")
    parser.add_argument("--anomaly-head", type=str,
                        default="output/seed_42_v2/anomaly_head.pt")
    parser.add_argument("--fusions", type=str, default="max,l2,sum_positive",
                        help="Comma-separated fusion strategies to test")
    parser.add_argument("--signals", type=str, default="v3,all",
                        help="Signal sets: v3 (4 signals), all (7-8 signals)")
    args = parser.parse_args()

    # Load V2 checkpoint
    model_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(model_dir, args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg, model_state, saved_proto = ckpt["config"], ckpt["model_state"], ckpt["prototypes"]
    device = cfg.DEVICE
    print(f"Device: {device}")

    # Load vocabs
    from vocabulary import OpcVocab, ParamVocab
    opc_vocab = OpcVocab.load(os.path.join(model_dir, "opc_vocab.pkl"))
    param_vocab = ParamVocab.load(os.path.join(model_dir, "param_vocab.pkl"))
    with open(os.path.join(model_dir, "atom_registry.pkl"), "rb") as f:
        atom_registry = pickle.load(f)

    from dataset import create_dataloaders
    loaders = create_dataloaders(cfg, opc_vocab, param_vocab, atom_registry)

    # Build model
    from model import VulnDetectionModel
    from embeddings import HybridEmbedding
    o_sz = model_state["embedding.opc_embed.weight"].shape[0]
    a_ct = model_state["embedding.param_embed.weight"].shape[0]
    o_d = model_state["embedding.opc_embed.weight"].shape[1]
    p_d = model_state["embedding.param_embed.weight"].shape[1]
    he = HybridEmbedding(cfg, np.zeros((o_sz, o_d), dtype=np.float32),
                         np.zeros((a_ct, p_d), dtype=np.float32))
    he.build_atom_lookup(param_vocab, atom_registry)
    model = VulnDetectionModel(cfg, cfg.NUM_KNOWN)
    model.set_embedding(he, atom_registry)
    model.load_state_dict(model_state, strict=False)
    model = model.to(device)
    model.eval()
    prototypes = saved_proto
    print(f"Loaded model from: {args.checkpoint}")

    # Load anomaly head
    from anomaly_head import AnomalyHead
    anomaly_head = AnomalyHead(input_dim=cfg.FC2_UNITS, hidden_dim=16)
    anomaly_head.load_state_dict(torch.load(
        os.path.join(model_dir, args.anomaly_head), map_location="cpu", weights_only=True))
    anomaly_head = anomaly_head.to(device)
    anomaly_head.eval()
    print(f"Loaded anomaly head from: {args.anomaly_head}")

    # ---- Step 1: Fit Mahalanobis stats (once) ----
    print("\n[1/4] Fitting Mahalanobis statistics...")
    mahal_stats = fit_mahalanobis_stats(model, loaders["train"], cfg, device)

    # ---- Step 2: Collect per-class stats for each signal set ----
    print("\n[2/4] Collecting per-class statistics...")

    # V3 stats (4 signals)
    from detect_v2 import collect_per_class_stats as collect_v3_stats
    pcs_v3 = collect_v3_stats(model, loaders["val"], prototypes, cfg, device)
    # Add anomaly stats
    from train_anomaly import collect_per_class_stats_v3
    pcs_v3_anom = collect_per_class_stats_v3(
        model, anomaly_head, loaders["val"], prototypes, cfg, device)

    # V4 all-signals stats
    pcs_v4 = collect_per_class_stats_v4(
        model, anomaly_head, loaders["val"], prototypes, mahal_stats, cfg, device)

    # ---- Step 3: Calibrate thresholds ----
    print("\n[3/4] Calibrating thresholds...")

    # All configurations to test
    configs = []
    for sig_set in args.signals.split(","):
        for fus in args.fusions.split(","):
            configs.append((sig_set.strip(), fus.strip()))

    results_table = []

    for sig_set, fusion in configs:
        print(f"\n--- {sig_set} signals, {fusion} fusion ---")

        # Choose per-class stats
        if sig_set == "v3":
            pcs = pcs_v3_anom if "anom" in str(pcs_v3_anom.get(0, {})) else pcs_v3
        else:
            pcs = pcs_v4

        # Calibrate
        thresh = calibrate_ood_threshold_v4(
            model, anomaly_head, loaders["val"], prototypes,
            mahal_stats, pcs, cfg, device,
            fusion=fusion, signals=sig_set)

        # Detect
        results = detect_v4(
            model, anomaly_head, loaders["test"], prototypes, mahal_stats,
            pcs, thresh, cfg, device,
            fusion=fusion, signals=sig_set)

        metrics = compute_unknown_metrics(results)
        tp = sum(1 for r in results if r["is_unknown"] and r["decision"] == "unknown")
        fp = sum(1 for r in results if not r["is_unknown"] and r["decision"] == "unknown")
        fn = sum(1 for r in results if r["is_unknown"] and r["decision"] == "known")

        row = {
            "signals": sig_set, "fusion": fusion, "threshold": thresh,
            "UDR": metrics["UDR"], "FPR": metrics["FPR"],
            "F1_unknown": metrics["F1_unknown"],
            "TP": tp, "FP": fp, "FN": fn,
        }
        results_table.append(row)

        print(f"  UDR={metrics['UDR']:.4f}  FPR={metrics['FPR']:.4f}  "
              f"F1={metrics['F1_unknown']:.4f}  TP={tp}  FP={fp}  FN={fn}")

    # ---- Step 4: Summary ----
    print("\n" + "=" * 90)
    print("[4/4] V3 vs V4 Comparison")
    print("=" * 90)
    print(f"{'Signals':<8s} {'Fusion':<14s} {'Thresh':>7s} {'UDR':>8s} {'FPR':>8s} {'F1':>8s} {'TP':>6s} {'FP':>6s} {'FN':>5s}")
    print("-" * 90)
    for row in sorted(results_table, key=lambda r: -r["UDR"]):
        print(f"{row['signals']:<8s} {row['fusion']:<14s} {row['threshold']:>7.2f} "
              f"{row['UDR']:>8.4f} {row['FPR']:>8.4f} {row['F1_unknown']:>8.4f} "
              f"{row['TP']:>6d} {row['FP']:>6d} {row['FN']:>5d}")

    # Find best
    best = max(results_table, key=lambda r: r["UDR"] - r["FPR"] * 0.5)
    print(f"\nBest (UDR - 0.5*FPR): {best['signals']} + {best['fusion']} "
          f"UDR={best['UDR']:.4f} FPR={best['FPR']:.4f}")

    # Best pure UDR
    best_udr = max(results_table, key=lambda r: r["UDR"])
    print(f"Best UDR: {best_udr['signals']} + {best_udr['fusion']} "
          f"UDR={best_udr['UDR']:.4f} FPR={best_udr['FPR']:.4f}")

    # Save best results
    out_dir = os.path.dirname(ckpt_path)
    json.dump(results_table, open(os.path.join(out_dir, "v4_comparison.json"), "w"), indent=2)
    print(f"\nSaved: {out_dir}/v4_comparison.json")


if __name__ == "__main__":
    main()
