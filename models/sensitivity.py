#!/usr/bin/env python3
"""
Hyperparameter Sensitivity Analysis.
Section 5.6.5 of the paper.

Analyses:
  1. tau_high × tau_low grid search → F1 heatmap
  2. MC Dropout T sensitivity → UDR / FPR curves
  3. L_max trade-off → performance vs memory/time
"""

import os
import json
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from detect import mc_dropout_forward, compute_prototype_distances
from evaluate import compute_unknown_metrics


def _build_fused(model, batch, device):
    opc = batch["opc_indices"].to(device)
    param = batch["param_indices"].to(device)
    stats = batch["stats"].to(device)
    return model.embedding(opc, param, stats, model.atom_registry)


def _collect_val_predictions(model, val_loader, prototypes, config, device):
    """Collect MC Dropout predictions on validation set (known-only)."""
    records = []
    for batch in val_loader:
        mc = mc_dropout_forward(model, batch, device, T=config.MC_DROPOUT_T)
        min_dist, _ = compute_prototype_distances(mc["penult_features"], prototypes)
        for i in range(batch["labels"].size(0)):
            if not batch["is_unknown"][i].item():
                records.append({
                    "max_prob": mc["max_prob"][i].item(),
                    "entropy": mc["entropy"][i].item(),
                    "min_dist": min_dist[i].item(),
                })
    return records


def grid_search_tau(val_records, config):
    """
    Grid search over tau_high × tau_low using pseudo-unknown sampling.
    Returns (best_tau_high, best_tau_low, f1_matrix, tau_high_vals, tau_low_vals).
    """
    n = len(val_records)
    n_pseudo = max(1, int(n * config.VAL_PSEUDO_UNKNOWN_FRAC))
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    normal_idx = indices[:n - n_pseudo]
    pseudo_idx = indices[n - n_pseudo:]

    normal_recs = [val_records[i] for i in normal_idx]
    pseudo_recs = [val_records[i] for i in pseudo_idx]

    H_low = np.percentile([m["entropy"] for m in normal_recs], 50)
    H_high = np.percentile([m["entropy"] for m in pseudo_recs], 75)
    delta_max = np.percentile([m["min_dist"] for m in normal_recs], 95)

    tau_high_vals = np.linspace(0.70, 0.95, 11)
    tau_low_vals = np.linspace(0.30, 0.60, 13)
    f1_matrix = np.zeros((len(tau_high_vals), len(tau_low_vals)))

    best_f1 = 0.0
    best_th = 0.85
    best_tl = 0.45

    for i, th in enumerate(tau_high_vals):
        for j, tl in enumerate(tau_low_vals):
            tp = 0
            fp = 0
            fn = 0
            for m in pseudo_recs:
                known = (m["max_prob"] >= th and m["entropy"] <= H_low
                         and m["min_dist"] <= delta_max)
                if not known:
                    tp += 1
                else:
                    fn += 1
            for m in normal_recs:
                known = (m["max_prob"] >= th and m["entropy"] <= H_low
                         and m["min_dist"] <= delta_max)
                if not known:
                    fp += 1

            prec = tp / (tp + fp + 1e-10)
            rec = tp / (tp + fn + 1e-10)
            f1 = 2 * prec * rec / (prec + rec + 1e-10)
            f1_matrix[i, j] = f1

            if f1 > best_f1:
                best_f1 = f1
                best_th = th
                best_tl = tl

    return {
        "best_tau_high": float(best_th),
        "best_tau_low": float(best_tl),
        "H_low": float(H_low),
        "H_high": float(H_high),
        "delta_max": float(delta_max),
        "best_f1": float(best_f1),
        "tau_high_vals": tau_high_vals.tolist(),
        "tau_low_vals": tau_low_vals.tolist(),
        "f1_matrix": f1_matrix.tolist(),
    }


def mc_dropout_sensitivity(model, val_loader, test_loader, prototypes, config, device):
    """
    Vary MC Dropout T ∈ {5, 10, 20, 30} and measure UDR, FPR, F1@unknown.
    """
    T_values = [5, 10, 20, 30]
    results = []

    for T in T_values:
        print(f"    MC Dropout T={T}...")
        det_results = _detect_with_T(model, test_loader, prototypes, config, T, device)
        metrics = compute_unknown_metrics(det_results)
        results.append({
            "T": T,
            "UDR": metrics["UDR"],
            "FPR": metrics["FPR"],
            "F1_unknown": metrics["F1_unknown"],
            "ROC_AUC": metrics["ROC_AUC"],
        })

    return results


def _detect_with_T(model, dataloader, prototypes, config, T, device):
    """Run detection with a specific MC Dropout T value."""
    results = []
    for batch in tqdm(dataloader, desc=f"      T={T}", leave=False):
        mc = mc_dropout_forward(model, batch, device, T=T)
        min_dist, nearest_class = compute_prototype_distances(
            mc["penult_features"], prototypes)

        B = batch["labels"].size(0)
        for i in range(B):
            mp = mc["max_prob"][i].item()
            ent = mc["entropy"][i].item()
            md = min_dist[i].item()

            known_cond = (mp >= config.TAU_HIGH and ent <= config.ENTROPY_LOW)
            unknown_cond = (mp < config.TAU_LOW or ent >= config.ENTROPY_HIGH)

            if known_cond:
                decision = "known"
            elif unknown_cond:
                decision = "unknown"
            else:
                decision = "ambiguous"

            results.append({
                "true_label": batch["labels"][i].item(),
                "is_unknown": batch["is_unknown"][i].item(),
                "category": batch.get("category", [""])[i],
                "basename": batch["basename"][i] if "basename" in batch else "",
                "max_prob": mp,
                "entropy": ent,
                "min_dist": md,
                "pred_class": mc["pred_class"][i].item(),
                "decision": decision,
            })

    return results


def lmax_sensitivity(config, opc_vocab, param_vocab, atom_registry,
                     loaders, device):
    """
    Vary L_max ∈ {2000, 5000, 10000} and evaluate performance vs memory/time.
    NOTE: This is a lighter-weight analysis — re-trains embeddings and model for each L_max.
    For full results paper-style, run the full pipeline with different L_max values.
    Here we measure memory and inference time on a sample batch.
    """
    from model import VulnDetectionModel

    L_values = [2000, 5000, 10000]
    results = []

    # Use a representative batch from the test set
    test_batch = next(iter(loaders["test"]))

    for L_max in L_values:
        print(f"    L_max={L_max}...")
        vconfig = copy.deepcopy(config)
        vconfig.L_MAX = L_max

        # Build a model with this L_max
        model = VulnDetectionModel(vconfig, vconfig.NUM_KNOWN)
        # Train a quick embedding
        from embeddings import train_all_embeddings
        hybrid_emb = train_all_embeddings(
            vconfig, opc_vocab, param_vocab, atom_registry, loaders["train"])
        model.set_embedding(hybrid_emb, atom_registry)
        model = model.to(device)
        model.eval()

        # Measure inference time
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in test_batch.items()}
        # Truncate to L_max
        actual_L = min(batch["mask"].sum(dim=1).max().item(), L_max)
        for key in ["opc_indices", "param_indices", "stats", "mask"]:
            batch[key] = batch[key][:, :actual_L]
        # Adjust edge_index
        ei = batch["edge_index"]
        valid = (ei[0] < actual_L) & (ei[1] < actual_L)
        batch["edge_index"] = ei[:, valid]

        # Warmup
        for _ in range(3):
            fused = _build_fused(model, batch, device)
            _ = model(fused, batch["edge_index"], batch["edge_attr"], batch["mask"])

        # Timed runs
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.time()
        n_runs = 50
        for _ in range(n_runs):
            fused = _build_fused(model, batch, device)
            _ = model(fused, batch["edge_index"], batch["edge_attr"], batch["mask"])
        torch.cuda.synchronize() if device == "cuda" else None
        avg_time_ms = (time.time() - t0) / n_runs * 1000 / batch["batch_size"]

        # Memory
        if device == "cuda":
            mem_mb = torch.cuda.max_memory_allocated() / 1024**2
            torch.cuda.reset_peak_memory_stats()
        else:
            mem_mb = 0

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())

        results.append({
            "L_max": L_max,
            "actual_L": int(actual_L),
            "avg_inference_ms": round(avg_time_ms, 3),
            "gpu_memory_mb": round(mem_mb, 1),
            "parameters": n_params,
        })

        print(f"      inference={avg_time_ms:.2f}ms/tx, mem={mem_mb:.1f}MB, params={n_params:,}")

    return results


def run_sensitivity_analysis(config, model, val_loader, test_loader,
                            prototypes, opc_vocab, param_vocab, atom_registry,
                            loaders, device):
    """Run all sensitivity analyses. Returns dict of results."""
    print("\n" + "="*60)
    print("HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("="*60)

    out_dir = os.path.join(config.OUTPUT_DIR, "sensitivity")
    os.makedirs(out_dir, exist_ok=True)

    # 1. tau grid search
    print("\n[1/3] Tau grid search...")
    val_records = _collect_val_predictions(model, val_loader, prototypes, config, device)
    tau_results = grid_search_tau(val_records, config)
    print(f"  Best tau_high={tau_results['best_tau_high']:.3f}, "
          f"tau_low={tau_results['best_tau_low']:.3f}, F1={tau_results['best_f1']:.4f}")

    with open(os.path.join(out_dir, "tau_grid.json"), 'w') as f:
        json.dump(tau_results, f, indent=2)

    # 2. MC Dropout T sensitivity
    print("\n[2/3] MC Dropout T sensitivity...")
    mc_results = mc_dropout_sensitivity(
        model, val_loader, test_loader, prototypes, config, device)
    for r in mc_results:
        print(f"  T={r['T']:2d}: UDR={r['UDR']:.4f}, FPR={r['FPR']:.4f}, F1={r['F1_unknown']:.4f}")

    with open(os.path.join(out_dir, "mc_dropout_sensitivity.json"), 'w') as f:
        json.dump(mc_results, f, indent=2)

    # 3. L_max sensitivity
    print("\n[3/3] L_max sensitivity...")
    lmax_results = lmax_sensitivity(
        config, opc_vocab, param_vocab, atom_registry, loaders, device)
    for r in lmax_results:
        print(f"  L_max={r['L_max']:5d}: {r['avg_inference_ms']:.2f}ms/tx, "
              f"{r['gpu_memory_mb']:.1f}MB, {r['parameters']:,} params")

    with open(os.path.join(out_dir, "lmax_sensitivity.json"), 'w') as f:
        json.dump(lmax_results, f, indent=2)

    # Generate plots
    plot_sensitivity_figures(tau_results, mc_results, lmax_results,
                             os.path.join(config.PLOT_DIR, "sensitivity"))

    return {
        "tau_grid": tau_results,
        "mc_dropout": mc_results,
        "lmax": lmax_results,
    }


def plot_sensitivity_figures(tau_results, mc_results, lmax_results, save_dir):
    """Generate sensitivity analysis figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13,
                         'figure.dpi': 150, 'savefig.dpi': 150,
                         'savefig.bbox': 'tight'})
    os.makedirs(save_dir, exist_ok=True)

    # Figure 1: Tau heatmap
    f1_matrix = np.array(tau_results["f1_matrix"])
    tau_high_vals = np.array(tau_results["tau_high_vals"])
    tau_low_vals = np.array(tau_results["tau_low_vals"])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(f1_matrix, aspect='auto', origin='lower',
                   extent=[tau_low_vals[0], tau_low_vals[-1],
                           tau_high_vals[0], tau_high_vals[-1]],
                   cmap='YlOrRd')
    ax.set_xlabel("τ_low")
    ax.set_ylabel("τ_high")
    ax.set_title("F1 Score vs Thresholds (Validation Pseudo-Unknown)")
    plt.colorbar(im, ax=ax, label='F1 Score')

    # Mark best point
    best_i = np.unravel_index(np.argmax(f1_matrix), f1_matrix.shape)
    ax.plot(tau_low_vals[best_i[1]], tau_high_vals[best_i[0]],
            'b*', markersize=20, markeredgecolor='white', markeredgewidth=1.5,
            label=f"Best (τ_h={tau_results['best_tau_high']:.2f}, "
                  f"τ_l={tau_results['best_tau_low']:.2f})")
    ax.legend()
    fig.savefig(os.path.join(save_dir, "tau_heatmap.png"))
    plt.close(fig)
    print(f"  Saved: {os.path.join(save_dir, 'tau_heatmap.png')}")

    # Figure 2: MC Dropout T sensitivity
    T_vals = [r["T"] for r in mc_results]
    udr_vals = [r["UDR"] for r in mc_results]
    fpr_vals = [r["FPR"] for r in mc_results]
    f1_vals = [r["F1_unknown"] for r in mc_results]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(T_vals, udr_vals, 'o-', color='#2ecc71', linewidth=2, markersize=8, label='UDR')
    ax.plot(T_vals, fpr_vals, 's-', color='#e74c3c', linewidth=2, markersize=8, label='FPR')
    ax.plot(T_vals, f1_vals, '^-', color='#3498db', linewidth=2, markersize=8, label='F1@unknown')
    ax.set_xlabel("MC Dropout Samples (T)")
    ax.set_ylabel("Score")
    ax.set_title("MC Dropout Sensitivity")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.savefig(os.path.join(save_dir, "mc_dropout_sensitivity.png"))
    plt.close(fig)
    print(f"  Saved: {os.path.join(save_dir, 'mc_dropout_sensitivity.png')}")

    # Figure 3: L_max trade-off (dual y-axis)
    L_vals = [r["L_max"] for r in lmax_results]
    time_vals = [r["avg_inference_ms"] for r in lmax_results]
    mem_vals = [r["gpu_memory_mb"] for r in lmax_results]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    color1 = '#3498db'
    ax1.set_xlabel("L_max (Sequence Length)")
    ax1.set_ylabel("Inference Time (ms/tx)", color=color1)
    ax1.plot(L_vals, time_vals, 'o-', color=color1, linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = '#e74c3c'
    ax2.set_ylabel("GPU Memory (MB)", color=color2)
    ax2.plot(L_vals, mem_vals, 's-', color=color2, linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title("L_max Trade-off: Inference Time vs Memory")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "lmax_tradeoff.png"))
    plt.close(fig)
    print(f"  Saved: {os.path.join(save_dir, 'lmax_tradeoff.png')}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import Config
    from vocabulary import build_all_vocabs
    from dataset import create_dataloaders

    cfg = Config()
    print(f"Device: {cfg.DEVICE_NAME}")
    opc_vocab, param_vocab, atom_registry = build_all_vocabs(cfg)
    loaders = create_dataloaders(cfg, opc_vocab, param_vocab, atom_registry)
    print("Sensitivity module loaded successfully.")
