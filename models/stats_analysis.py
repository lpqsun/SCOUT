#!/usr/bin/env python3
"""
Dataset statistics and computational efficiency analysis.
Sections 5.2 (Table 1) and 5.6.6 of the paper.

Generates:
  - Table 1: Per-class dataset statistics
  - Efficiency metrics: training/inference time, GPU memory
"""

import os
import json
import time
import numpy as np
from collections import defaultdict


def compute_dataset_statistics(config):
    """
    Compute per-category dataset statistics for Table 1.
    Returns structured dict with per-category and aggregate stats.
    """
    base_dir = config.DATA_ROOT
    stats = {}

    for category in config.CATEGORIES:
        param_dir = os.path.join(base_dir, category + config.PARAM_DIR_SUFFIX)
        if not os.path.isdir(param_dir):
            continue

        lengths = []
        opcode_counts = defaultdict(int)
        param_type_counts = defaultdict(int)
        n_traces = 0

        # List param_only.txt files
        param_files = sorted([
            f for f in os.listdir(param_dir)
            if f.endswith("_param_only.txt")
        ])

        for pf in param_files:
            fpath = os.path.join(param_dir, pf)
            try:
                with open(fpath, 'r') as f:
                    tokens = [line.strip() for line in f if line.strip()]
                    lengths.append(len(tokens))
                n_traces += 1
            except Exception:
                continue

            # Also read opcodes from param_features.json
            json_path = fpath.replace("_param_only.txt", "_param_features.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    for step in data.get("steps", []):
                        opcode_counts[step.get("opcode", "UNK")] += 1
                except Exception:
                    pass

        if n_traces == 0:
            continue

        lengths_arr = np.array(lengths)
        role = "Pseudo-Unknown" if category in config.PSEUDO_UNKNOWN_CLASSES else "Known"

        stats[category] = {
            "role": role,
            "n_traces": n_traces,
            "total_length": int(lengths_arr.sum()),
            "mean_length": float(lengths_arr.mean()),
            "median_length": float(np.median(lengths_arr)),
            "min_length": int(lengths_arr.min()),
            "max_length": int(lengths_arr.max()),
            "p25_length": float(np.percentile(lengths_arr, 25)),
            "p75_length": float(np.percentile(lengths_arr, 75)),
            "p90_length": float(np.percentile(lengths_arr, 90)),
            "gt_5000_pct": float((lengths_arr > 5000).mean() * 100),
            "unique_opcodes": len(opcode_counts),
            "unique_param_types": len(param_type_counts),
        }

    # Known aggregate
    known_cats = [c for c in config.CATEGORIES if c in config.KNOWN_CLASSES and c in stats]
    unknown_cats = [c for c in config.CATEGORIES if c in config.PSEUDO_UNKNOWN_CLASSES and c in stats]

    all_lengths = []
    for cat in stats:
        cat_stats = stats[cat]
        # Recompute all lengths for global stats
        param_dir = os.path.join(base_dir, cat + config.PARAM_DIR_SUFFIX)
        for pf in os.listdir(param_dir):
            if pf.endswith("_param_only.txt"):
                try:
                    with open(os.path.join(param_dir, pf), 'r') as f:
                        all_lengths.append(sum(1 for line in f if line.strip()))
                except Exception:
                    pass

    all_arr = np.array(all_lengths) if all_lengths else np.array([0])

    aggregate = {
        "total_traces": int(len(all_lengths)),
        "known_traces": sum(stats[c]["n_traces"] for c in known_cats),
        "unknown_traces": sum(stats[c]["n_traces"] for c in unknown_cats),
        "mean_length": float(all_arr.mean()),
        "median_length": float(np.median(all_arr)),
        "min_length": int(all_arr.min()),
        "max_length": int(all_arr.max()),
        "p90_length": float(np.percentile(all_arr, 90)),
        "gt_5000_pct": float((all_arr > 5000).mean() * 100),
        "num_known_classes": len(known_cats),
        "num_unknown_classes": len(unknown_cats),
    }

    # Train/Val/Test split info
    split_info = {}
    for cat in known_cats:
        n = stats[cat]["n_traces"]
        split_info[cat] = {
            "total": n,
            "train": int(n * config.TRAIN_RATIO),
            "val": int(n * config.VAL_RATIO),
            "test": int(n * config.TEST_RATIO),
        }

    return {
        "per_category": stats,
        "aggregate": aggregate,
        "split_info": split_info,
        "known_classes": list(known_cats),
        "unknown_classes": list(unknown_cats),
    }


def print_dataset_table(stats):
    """Pretty-print Table 1: Dataset statistics."""
    print("\n" + "="*100)
    print("TABLE 1: Dataset Statistics")
    print("="*100)

    header = (f"{'Category':<30} {'Role':>14} {'N':>8} {'Mean L':>8} "
              f"{'Median L':>8} {'Min L':>6} {'Max L':>8} {'P90 L':>8} {'%>5k':>6}")
    print(header)
    print("-"*100)

    for cat, s in stats["per_category"].items():
        cat_name = cat.replace("_", " ").title()
        print(f"{cat_name:<30} {s['role']:>14} {s['n_traces']:8d} {s['mean_length']:8.1f} "
              f"{s['median_length']:8.1f} {s['min_length']:6d} {s['max_length']:8d} "
              f"{s['p90_length']:8.1f} {s['gt_5000_pct']:5.1f}%")

    print("-"*100)
    agg = stats["aggregate"]
    print(f"{'AGGREGATE':<30} {'':>14} {agg['total_traces']:8d} {agg['mean_length']:8.1f} "
          f"{agg['median_length']:8.1f} {agg['min_length']:6d} {agg['max_length']:8d} "
          f"{agg['p90_length']:8.1f} {agg['gt_5000_pct']:5.1f}%")

    print(f"\n  Known classes ({agg['num_known_classes']}): {stats['known_classes']}")
    print(f"  Unknown/Pseudo-unknown classes ({agg['num_unknown_classes']}): {stats['unknown_classes']}")
    print(f"  Train/Val/Test split: {config.TRAIN_RATIO:.0%}/{config.VAL_RATIO:.0%}/{config.TEST_RATIO:.0%}")

    # Per-class split sizes
    print(f"\n  {'Class':<30} {'Total':>8} {'Train':>8} {'Val':>8} {'Test':>8}")
    for cat, sp in stats["split_info"].items():
        cat_name = cat.replace("_", " ").title()
        print(f"  {cat_name:<30} {sp['total']:8d} {sp['train']:8d} {sp['val']:8d} {sp['test']:8d}")


def measure_efficiency(config, model, test_loader, device, n_warmup=10, n_measure=100):
    """
    Measure computational efficiency metrics (Section 5.6.6).
    Returns dict with training and inference timing.
    """
    import torch

    print("\n  Measuring inference efficiency...")
    model.eval()

    # Get a single sample batch
    batch = next(iter(test_loader))
    B = batch["batch_size"]

    # Move to device
    batch_gpu = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    # Warmup
    for _ in range(n_warmup):
        fused = model.embedding(
            batch_gpu["opc_indices"], batch_gpu["param_indices"],
            batch_gpu["stats"], model.atom_registry)
        _ = model(fused, batch_gpu["edge_index"], batch_gpu["edge_attr"],
                  batch_gpu["mask"])

    # Synchronize and time
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_measure):
        fused = model.embedding(
            batch_gpu["opc_indices"], batch_gpu["param_indices"],
            batch_gpu["stats"], model.atom_registry)
        _ = model(fused, batch_gpu["edge_index"], batch_gpu["edge_attr"],
                  batch_gpu["mask"])
    if device == "cuda":
        torch.cuda.synchronize()
    total_time = time.time() - t0

    avg_inference_ms = (total_time / n_measure) * 1000 / B

    # Memory
    if device == "cuda":
        mem_allocated = torch.cuda.max_memory_allocated() / 1024**2
        mem_reserved = torch.cuda.max_memory_reserved() / 1024**2
        torch.cuda.reset_peak_memory_stats()
    else:
        mem_allocated = 0
        mem_reserved = 0

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Model size on disk (approx)
    param_size_mb = n_params * 4 / 1024**2  # float32

    results = {
        "avg_inference_ms_per_tx": round(avg_inference_ms, 3),
        "batch_size": int(B),
        "n_warmup": n_warmup,
        "n_measure": n_measure,
        "gpu_memory_allocated_mb": round(mem_allocated, 1),
        "gpu_memory_reserved_mb": round(mem_reserved, 1),
        "total_parameters": n_params,
        "trainable_parameters": n_trainable,
        "model_size_mb": round(param_size_mb, 1),
        "device": device,
        "model_architecture": {
            "cnn_filters": config.CNN_FILTERS,
            "bilstm_units": config.BILSTM_UNITS,
            "gcn_hidden": config.GCN_HIDDEN,
            "gcn_out": config.GCN_OUT,
            "attention_heads": config.ATTENTION_HEADS,
            "fc1_units": config.FC1_UNITS,
            "fc2_units": config.FC2_UNITS,
            "fused_dim": config.FUSED_DIM,
        },
    }

    return results


def print_efficiency_table(eff):
    """Pretty-print computational efficiency results."""
    print("\n" + "="*70)
    print("COMPUTATIONAL EFFICIENCY (Section 5.6.6)")
    print("="*70)
    print(f"  Device: {eff['device']}")
    print(f"  Model parameters: {eff['total_parameters']:,} "
          f"({eff['trainable_parameters']:,} trainable)")
    print(f"  Model size (approx): {eff['model_size_mb']:.1f} MB")
    print(f"  GPU memory allocated: {eff['gpu_memory_allocated_mb']:.1f} MB")
    print(f"  GPU memory reserved:  {eff['gpu_memory_reserved_mb']:.1f} MB")
    print(f"  Avg inference time:   {eff['avg_inference_ms_per_tx']:.2f} ms/tx")
    print(f"  Throughput:            {1000/eff['avg_inference_ms_per_tx']:.1f} tx/s")
    print(f"  (Batch size={eff['batch_size']}, "
          f"measured over {eff['n_measure']} runs after {eff['n_warmup']} warmup)")


def run_stats_and_efficiency(config, model, test_loader, device):
    """Run full statistics and efficiency analysis."""
    out_dir = os.path.join(config.OUTPUT_DIR, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    # Dataset statistics
    print("\nComputing dataset statistics...")
    ds_stats = compute_dataset_statistics(config)
    print_dataset_table(ds_stats)

    with open(os.path.join(out_dir, "dataset_statistics.json"), 'w') as f:
        json.dump(ds_stats, f, indent=2)

    # Computational efficiency
    eff = measure_efficiency(config, model, test_loader, device)
    print_efficiency_table(eff)

    with open(os.path.join(out_dir, "efficiency.json"), 'w') as f:
        json.dump(eff, f, indent=2)

    # Combined report
    return {
        "dataset_statistics": ds_stats,
        "efficiency": eff,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import Config

    cfg = Config()
    ds_stats = compute_dataset_statistics(cfg)
    print_dataset_table(ds_stats)
