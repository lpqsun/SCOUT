#!/usr/bin/env python3
"""
Full experiment runner for smart contract vulnerability detection.
- Builds vocabularies
- Precomputes statistical features
- Trains CBOW embeddings
- Trains the CNN-BiLSTM-Attention-GCN model
- Evaluates known-class classification and unknown-vulnerability detection
- Runs 7 ablation variants
- Generates all plots

Usage:
  python run_experiments.py                 # full pipeline
  python run_experiments.py --quick         # quick test on 1000 samples
  python run_experiments.py --ablation-only # only ablation after training
"""

import os
import sys
import json
import time
import copy
import pickle
import argparse
import numpy as np
import torch

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from vocabulary import build_all_vocabs, OpcVocab, ParamVocab, AtomicRegistry
from precompute import precompute_all
from dataset import create_dataloaders
from embeddings import train_all_embeddings
from model import VulnDetectionModel, build_model_variant
from train import train
from detect import detect_unknown, calibrate_thresholds
from evaluate import (
    compute_known_metrics, compute_unknown_metrics,
    print_known_metrics, print_unknown_metrics,
)
from plot import (
    plot_confusion_matrix, plot_roc_curves, plot_ablation_bars,
    plot_tsne, plot_perf_vs_length, plot_attention_heatmap,
    plot_training_history, plot_pr_curve, plot_multi_roc_curves,
    plot_baseline_comparison, plot_per_unknown_type,
    plot_robustness_length, plot_decision_distribution,
)
from sensitivity import run_sensitivity_analysis
from stats_analysis import run_stats_and_efficiency
from baselines import run_baseline_comparison
from stage1_svdd import DeepSVDD, TwoStageDetector, evaluate_stage1


def run_single_experiment(config, seed=42, quick=False, skip_precompute=False):
    """Run a complete experiment with one seed. Returns all results dict."""
    device = config.DEVICE
    print(f"  Using device: {config.DEVICE_NAME}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    out_dir = os.path.join(config.OUTPUT_DIR, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENT — Seed {seed}")
    print(f"{'='*60}")

    # Step 1: Build vocabularies
    print("\n[1/5] Building vocabularies...")
    opc_vocab, param_vocab, atom_registry = build_all_vocabs(config)
    print(f"  OpcVocab: {len(opc_vocab)} tokens")
    print(f"  ParamVocab: {len(param_vocab)} tokens")
    print(f"  Atomic types: {len(atom_registry)}")

    # Step 2: Precompute statistical features
    if not skip_precompute:
        print("\n[2/5] Precomputing statistical features...")
        precompute_all(config)

    # Quick mode adjustments (must happen BEFORE dataloaders are created)
    if quick:
        config.EPOCHS = 15
        config.BATCH_SIZE = 4
        config.L_MAX = 500
        print("  (Quick mode: L_max=500, batch=4, epochs=15)")

    # Step 3: Create dataloaders
    print("\n[3/5] Creating dataloaders...")
    loaders = create_dataloaders(config, opc_vocab, param_vocab, atom_registry)

    # Quick mode: downsample datasets
    if quick:
        def downsample(loader, n):
            ds = loader.dataset
            ds.samples = ds.samples[:min(n, len(ds.samples))]
            return loader
        loaders["train"] = downsample(loaders["train"], 600)
        loaders["val"] = downsample(loaders["val"], 150)
        loaders["test"] = downsample(loaders["test"], 250)

    # Step 4: Train embeddings
    print("\n[4/5] Training embeddings...")
    hybrid_emb = train_all_embeddings(config, opc_vocab, param_vocab,
                                      atom_registry, loaders["train"])

    # Step 5: Train model
    print("\n[5/5] Training model...")
    model = VulnDetectionModel(config, config.NUM_KNOWN)
    model.set_embedding(hybrid_emb, atom_registry)

    best_state, history, prototypes = train(
        config, model, loaders["train"], loaders["val"], seed=seed, device=device)
    model.load_state_dict(best_state)
    model.eval()

    # Save
    torch.save({
        "model_state": best_state,
        "config": config,
        "prototypes": prototypes,
        "history": history,
    }, os.path.join(out_dir, "checkpoint_best.pt"))
    with open(os.path.join(out_dir, "training_metrics.json"), 'w') as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(out_dir, "vocab_info.json"), 'w') as f:
        json.dump({"opc_vocab_size": len(opc_vocab), "param_vocab_size": len(param_vocab)},
                  f, indent=2)

    # Plot training curves
    plot_training_history(history, os.path.join(out_dir, "training_curves.png"))

    # ── Evaluation ──
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    # Known-class evaluation
    print("\n  Known-class evaluation...")
    model.eval()
    all_known_labels = []
    all_known_preds = []
    all_penult = []
    all_lengths = []
    with torch.no_grad():
        for batch in loaders["test"]:
            mask = batch["mask"]
            labels = batch["labels"]
            # Only evaluate known samples
            known_mask = ~batch["is_unknown"]
            if known_mask.sum() == 0:
                continue
            known_idx = known_mask.nonzero(as_tuple=True)[0]
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device)
            fused = hybrid_emb(
                batch["opc_indices"].to(device),
                batch["param_indices"].to(device),
                batch["stats"].to(device),
                atom_registry,
            )
            output = model(fused, edge_index, edge_attr, mask.to(device))
            preds = output["logits"].argmax(dim=1).cpu()
            for idx in known_idx.tolist():
                all_known_labels.append(labels[idx].item())
                all_known_preds.append(preds[idx].item())
                all_penult.append(output["penult_features"][idx].cpu().numpy())
                all_lengths.append(mask[idx].sum().item())

    known_metrics = compute_known_metrics(
        all_known_labels, all_known_preds, config.NUM_KNOWN,
        class_names=config.known_class_names,
    )
    print_known_metrics(known_metrics)
    with open(os.path.join(out_dir, "known_metrics.json"), 'w') as f:
        # Convert numpy types for JSON
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in known_metrics.items()}, f, indent=2)

    # Plot confusion matrix
    plot_confusion_matrix(
        known_metrics["confusion_matrix"],
        config.known_class_names,
        os.path.join(config.PLOT_DIR, f"confusion_matrix_s{seed}.png"),
    )

    # t-SNE
    if len(all_penult) > 10:
        all_penult = np.stack(all_penult)
        plot_tsne(all_penult, all_known_labels,
                  config.known_class_names,
                  os.path.join(config.PLOT_DIR, f"tsne_s{seed}.png"))

    # Performance vs length
    bins = [(0, 100), (100, 300), (300, 500), (500, 1000), (1000, 5000)]
    f1_per_bin = []
    for lo, hi in bins:
        idxs = [i for i, l in enumerate(all_lengths) if lo <= l < hi]
        if idxs:
            bin_preds = [all_known_preds[i] for i in idxs]
            bin_labels = [all_known_labels[i] for i in idxs]
            cm = np.zeros((config.NUM_KNOWN, config.NUM_KNOWN))
            for t, p in zip(bin_labels, bin_preds):
                cm[t, p] += 1
            total = cm.sum()
            if total > 0:
                tp = np.diag(cm).sum()
                prec = tp / total
                rec = tp / (cm.sum(axis=1).sum() + 1e-10)
                f1 = 2 * prec * rec / (prec + rec + 1e-10)
                f1_per_bin.append(f1)
            else:
                f1_per_bin.append(0)
        else:
            f1_per_bin.append(0)
    plot_perf_vs_length(bins, f1_per_bin,
                        os.path.join(config.PLOT_DIR, f"perf_vs_length_s{seed}.png"))

    # ── Unknown Detection ──
    print("\n  Unknown vulnerability detection...")
    tau_high, tau_low, H_low, H_high, delta_max = calibrate_thresholds(
        model, loaders["val"], prototypes, config, device)
    detection_results = detect_unknown(
        model, loaders["test"], prototypes, config,
        tau_high=tau_high, tau_low=tau_low,
        H_low=H_low, H_high=H_high, delta_max=delta_max, device=device)

    unknown_metrics = compute_unknown_metrics(detection_results)
    print_unknown_metrics(unknown_metrics)
    with open(os.path.join(out_dir, "unknown_metrics.json"), 'w') as f:
        json.dump(unknown_metrics, f, indent=2)
    with open(os.path.join(out_dir, "detection_results.json"), 'w') as f:
        json.dump(detection_results, f, indent=2)

    # ROC curve data
    y_true = np.array([r["is_unknown"] for r in detection_results])
    y_scores = np.array([1.0 - r["max_prob"] for r in detection_results])
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        plot_roc_curves(
            [(fpr, tpr, unknown_metrics["ROC_AUC"])],
            ["Full Model"],
            os.path.join(config.PLOT_DIR, f"roc_curve_s{seed}.png"),
        )
    except Exception:
        pass

    # PR curve (Section 5.4)
    try:
        plot_pr_curve(y_true, y_scores,
                      os.path.join(config.PLOT_DIR, f"pr_curve_s{seed}.png"))
    except Exception:
        pass

    # Decision distribution plot
    try:
        plot_decision_distribution(detection_results,
                                   os.path.join(config.PLOT_DIR, f"decision_dist_s{seed}.png"))
    except Exception:
        pass

    # ── Robustness Analysis (Section 5.6.4) ──
    print("\n  Robustness analysis...")
    robustness_results = None
    try:
        robustness_results = _run_robustness_analysis(
            detection_results, all_lengths, all_known_labels, all_known_preds,
            config, seed)
    except Exception as e:
        print(f"  Robustness analysis skipped: {e}")

    # ── Baseline Comparison (Section 5.3) ──
    print("\n  Baseline comparison...")
    baseline_results = None
    try:
        baseline_results = run_baseline_comparison(
            model, loaders, prototypes, config, device)
        # Save
        baseline_save = {}
        for name, b in baseline_results.items():
            baseline_save[name] = {
                "metrics": b.get("metrics", {}),
                "error": b.get("error"),
            }
        with open(os.path.join(out_dir, "baseline_comparison.json"), 'w') as f:
            json.dump(baseline_save, f, indent=2)

        # Multi-ROC plot
        roc_data = {}
        for name, b in baseline_results.items():
            results_list = b.get("results", [])
            if results_list:
                yt = np.array([r["is_unknown"] for r in results_list])
                ys = np.array([1.0 - r["max_prob"] for r in results_list])
                roc_data[name] = {"y_true": yt, "y_scores": ys}
        if roc_data:
            plot_multi_roc_curves(
                roc_data,
                os.path.join(config.PLOT_DIR, f"baseline_roc_s{seed}.png"),
            )
        # Baseline bar chart
        baseline_metrics_dict = {
            name: b.get("metrics", {})
            for name, b in baseline_results.items()
            if b.get("metrics")
        }
        if baseline_metrics_dict:
            plot_baseline_comparison(
                baseline_metrics_dict,
                os.path.join(config.PLOT_DIR, f"baseline_comparison_s{seed}.png"),
            )
    except Exception as e:
        print(f"  Baseline comparison skipped: {e}")

    # Attention heatmap (first test sample)
    try:
        batch = next(iter(loaders["test"]))
        fused = hybrid_emb(
            batch["opc_indices"].to(device),
            batch["param_indices"].to(device),
            batch["stats"].to(device),
            atom_registry,
        )
        output = model(fused, batch["edge_index"].to(device),
                       batch["edge_attr"].to(device), batch["mask"].to(device))
        attn = output["attention_weights"][0].cpu().numpy()
        opcs = [opc_vocab.idx_to_token.get(i.item(), "?") for i in batch["opc_indices"][0][:200]]
        plot_attention_heatmap(attn, opcs,
                               os.path.join(config.PLOT_DIR, f"attention_s{seed}.png"))
    except Exception:
        pass

    # ── Sensitivity Analysis (Section 5.6.5) ──
    print("\n  Sensitivity analysis...")
    sensitivity_results = None
    try:
        sensitivity_results = run_sensitivity_analysis(
            config, model, loaders["val"], loaders["test"],
            prototypes, opc_vocab, param_vocab, atom_registry,
            loaders, device)
    except Exception as e:
        print(f"  Sensitivity analysis skipped: {e}")

    # ── Stats & Efficiency (Sections 5.2 & 5.6.6) ──
    print("\n  Computing dataset statistics & efficiency...")
    eff_results = None
    try:
        eff_results = run_stats_and_efficiency(config, model, loaders["test"], device)
    except Exception as e:
        print(f"  Stats/efficiency analysis skipped: {e}")

    # Compile results
    result = {
        "seed": seed,
        "known": known_metrics,
        "unknown": unknown_metrics,
        "thresholds": {
            "tau_high": tau_high, "tau_low": tau_low,
            "H_low": H_low, "H_high": H_high,
            "delta_max": delta_max,
        },
    }
    if sensitivity_results is not None:
        result["sensitivity"] = {k: v for k, v in sensitivity_results.items()
                                 if k != "tau_grid"}
    if eff_results is not None:
        result["dataset_stats"] = eff_results["dataset_statistics"]
        result["efficiency"] = eff_results["efficiency"]
    if robustness_results is not None:
        result["robustness"] = robustness_results
    if baseline_results is not None:
        result["baselines"] = {
            name: {"metrics": b.get("metrics", {})}
            for name, b in baseline_results.items()
        }
    with open(os.path.join(out_dir, "summary.json"), 'w') as f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in result.items()}, f, indent=2)

    return result


def _run_robustness_analysis(detection_results, all_lengths,
                             all_known_labels, all_known_preds,
                             config, seed):
    """
    Robustness analysis (Section 5.6.4):
      - Per-unknown-type UDR
      - Performance by sequence length bucket
    """
    out_dir = config.PLOT_DIR

    # ── Per-unknown-type analysis ──
    per_type = {}
    for r in detection_results:
        cat = r.get("category", "unknown")
        if not r["is_unknown"]:
            continue
        if cat not in per_type:
            per_type[cat] = {"tp": 0, "total": 0}
        per_type[cat]["total"] += 1
        if r["decision"] == "unknown":
            per_type[cat]["tp"] += 1

    per_type_metrics = {}
    for cat, counts in per_type.items():
        udr = counts["tp"] / counts["total"] if counts["total"] > 0 else 0
        per_type_metrics[cat] = {"UDR": udr, "count": counts["total"]}
        print(f"    {cat}: UDR={udr:.4f} ({counts['tp']}/{counts['total']})")

    if per_type_metrics:
        plot_per_unknown_type(
            per_type_metrics,
            os.path.join(out_dir, f"robustness_per_type_s{seed}.png"),
        )

    # ── Length robustness ──
    # Attach length info to detection results
    for i, r in enumerate(detection_results):
        if i < len(all_lengths):
            r["length"] = all_lengths[i]

    plot_robustness_length(
        detection_results,
        os.path.join(out_dir, f"robustness_length_s{seed}.png"),
    )

    return {
        "per_unknown_type": per_type_metrics,
    }


def run_ablation(config, opc_vocab, param_vocab, atom_registry, loaders,
                 variant_name, variant_label, device="cuda"):
    """Run a single ablation variant. Returns metrics dict."""
    print(f"\n{'='*50}")
    print(f"Ablation: {variant_label}")
    print(f"{'='*50}")

    model = build_model_variant(config, config.NUM_KNOWN, variant_name)
    # Build embeddings using the model's actual config (may have USE_PARAM_FEATURES/USE_STATS_FEATURES
    # disabled by the variant) so that HybridEmbedding output dim matches model input dim.
    model_config = model.config
    hybrid_emb = train_all_embeddings(model_config, opc_vocab, param_vocab,
                                      atom_registry, loaders["train"])
    model.set_embedding(hybrid_emb, atom_registry)

    best_state, history, prototypes = train(
        model_config, model, loaders["train"], loaders["val"], seed=42, device=device)
    model.load_state_dict(best_state)
    model.eval()

    # Known metrics
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for batch in loaders["test"]:
            known_mask = ~batch["is_unknown"]
            if known_mask.sum() == 0:
                continue
            known_idx = known_mask.nonzero(as_tuple=True)[0]
            fused = hybrid_emb(
                batch["opc_indices"].to(device), batch["param_indices"].to(device),
                batch["stats"].to(device), atom_registry)
            output = model(fused, batch["edge_index"].to(device),
                           batch["edge_attr"].to(device), batch["mask"].to(device))
            preds = output["logits"].argmax(dim=1).cpu()
            for idx in known_idx.tolist():
                all_labels.append(batch["labels"][idx].item())
                all_preds.append(preds[idx].item())

    known_m = compute_known_metrics(all_labels, all_preds, config.NUM_KNOWN)

    # Unknown metrics
    tau_high, tau_low, H_low, H_high, delta_max = calibrate_thresholds(
        model, loaders["val"], prototypes, config, device)
    det_results = detect_unknown(model, loaders["test"], prototypes, config,
                                 tau_high=tau_high, tau_low=tau_low,
                                 H_low=H_low, H_high=H_high,
                                 delta_max=delta_max, device=device)
    unknown_m = compute_unknown_metrics(det_results)

    return {
        "F1": known_m["weighted_f1"],
        "UDR": unknown_m["UDR"],
        "FPR": unknown_m["FPR"],
        "known": known_m,
        "unknown": unknown_m,
    }


def _save_ablation_results(results, ablation_dir):
    """Atomically save ablation results to disk (incremental checkpoint)."""
    os.makedirs(ablation_dir, exist_ok=True)
    tmp_path = os.path.join(ablation_dir, "results.json.tmp")
    final_path = os.path.join(ablation_dir, "results.json")
    with open(tmp_path, 'w') as f:
        json.dump(results, f, indent=2)
    os.replace(tmp_path, final_path)  # atomic rename


def run_all_ablations(config, opc_vocab, param_vocab, atom_registry, loaders):
    """Run all 7 ablation variants, with incremental saving and resume support."""
    print("\n" + "="*60)
    print("ABLATION STUDY")
    print("="*60)
    device = config.DEVICE

    ablation_dir = os.path.join(config.OUTPUT_DIR, "ablation")
    os.makedirs(ablation_dir, exist_ok=True)

    # Resume from existing results if available
    results = {}
    results_path = os.path.join(ablation_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        print(f"[Resume] Loaded {len(results)} existing results: {list(results.keys())}")

    variants = [
        ("full", "Full Model"),
        ("no_param", "No Param Features"),
        ("no_gnn", "No GNN Branch"),
        ("no_stats", "No Statistical Features"),
        ("single_head", "Single-Head Attention"),
    ]

    for vname, vlabel in variants:
        if vlabel in results:
            print(f"\n[Ablation] {vlabel} — already done, skipping")
            continue
        vconfig = copy.deepcopy(config)
        results[vlabel] = run_ablation(
            vconfig, opc_vocab, param_vocab, atom_registry, loaders,
            vname, vlabel)
        _save_ablation_results(results, ablation_dir)

    # For "no prototype distance" and "no MC dropout" — handled in detect.py
    # Run detection with modified config
    NOPROTO_KEY = "No Proto Distance"
    if NOPROTO_KEY not in results:
        print(f"\n{'='*50}")
        print(f"Ablation: {NOPROTO_KEY}")
        print(f"{'='*50}")
        vconfig = copy.deepcopy(config)
        vconfig.PROTOTYPE_LAMBDA = 0.0
        hybrid_emb = train_all_embeddings(vconfig, opc_vocab, param_vocab, atom_registry, loaders["train"])
        model = build_model_variant(vconfig, vconfig.NUM_KNOWN, "full")
        model.set_embedding(hybrid_emb, atom_registry)
        best_state, history, prototypes = train(vconfig, model, loaders["train"], loaders["val"], seed=42, device=device)
        model.load_state_dict(best_state)
        model.eval()
        # Calibrate thresholds (with None prototypes, distance check is skipped)
        tau_h_np, tau_l_np, H_l_np, H_h_np, delta_np = calibrate_thresholds(
            model, loaders["val"], prototypes, vconfig, device)
        det_r = detect_unknown(model, loaders["test"], None, vconfig,
                               tau_high=tau_h_np, tau_low=tau_l_np,
                               H_low=H_l_np, H_high=H_h_np, delta_max=None, device=device)
        all_l = []; all_p = []
        with torch.no_grad():
            for batch in loaders["test"]:
                known_mask = ~batch["is_unknown"]
                if known_mask.sum() == 0: continue
                known_idx = known_mask.nonzero(as_tuple=True)[0]
                fused = hybrid_emb(batch["opc_indices"].to(device), batch["param_indices"].to(device), batch["stats"].to(device), atom_registry)
                out = model(fused, batch["edge_index"].to(device), batch["edge_attr"].to(device), batch["mask"].to(device))
                preds = out["logits"].argmax(dim=1)
                for idx in known_idx.tolist():
                    all_l.append(batch["labels"][idx].item()); all_p.append(preds[idx].item())
        km = compute_known_metrics(all_l, all_p, config.NUM_KNOWN)
        um = compute_unknown_metrics(det_r)
        results[NOPROTO_KEY] = {"F1": km["weighted_f1"], "UDR": um["UDR"], "FPR": um["FPR"]}
        _save_ablation_results(results, ablation_dir)
    else:
        print(f"\n[Ablation] {NOPROTO_KEY} — already done, skipping")

    NOMC_KEY = "No MC Dropout"
    if NOMC_KEY not in results:
        print(f"\n{'='*50}")
        print(f"Ablation: {NOMC_KEY} (T=1, single softmax)")
        print(f"{'='*50}")
        # Train a fresh full model for fair comparison (MC Dropout only affects inference)
        vconfig2 = copy.deepcopy(config)
        hybrid_emb2 = train_all_embeddings(vconfig2, opc_vocab, param_vocab, atom_registry, loaders["train"])
        model2 = build_model_variant(vconfig2, vconfig2.NUM_KNOWN, "full")
        model2.set_embedding(hybrid_emb2, atom_registry)
        best_state2, history2, prototypes2 = train(vconfig2, model2, loaders["train"], loaders["val"], seed=42, device=device)
        model2.load_state_dict(best_state2)
        model2.eval()
        # Recompute known metrics for this model
        all_l2 = []; all_p2 = []
        with torch.no_grad():
            for batch in loaders["test"]:
                known_mask = ~batch["is_unknown"]
                if known_mask.sum() == 0: continue
                known_idx = known_mask.nonzero(as_tuple=True)[0]
                fused = hybrid_emb2(batch["opc_indices"].to(device), batch["param_indices"].to(device), batch["stats"].to(device), atom_registry)
                out = model2(fused, batch["edge_index"].to(device), batch["edge_attr"].to(device), batch["mask"].to(device))
                preds = out["logits"].argmax(dim=1)
                for idx in known_idx.tolist():
                    all_l2.append(batch["labels"][idx].item()); all_p2.append(preds[idx].item())
        km2 = compute_known_metrics(all_l2, all_p2, config.NUM_KNOWN)
        # Calibrate thresholds for this model
        tau_h2, tau_l2, H_l2, H_h2, delta2 = calibrate_thresholds(model2, loaders["val"], prototypes2, vconfig2, device)
        vconfig2.MC_DROPOUT_T = 1
        det_r2 = detect_unknown(model2, loaders["test"], prototypes2, vconfig2,
                                tau_high=tau_h2, tau_low=tau_l2,
                                H_low=H_l2, H_high=H_h2, delta_max=delta2, device=device)
        um2 = compute_unknown_metrics(det_r2)
        results[NOMC_KEY] = {"F1": km2["weighted_f1"], "UDR": um2["UDR"], "FPR": um2["FPR"]}
        _save_ablation_results(results, ablation_dir)
    else:
        print(f"\n[Ablation] {NOMC_KEY} — already done, skipping")

    # Final save (already saved incrementally; this ensures latest + triggers plot)
    plot_ablation_bars(results,
                       os.path.join(config.PLOT_DIR, "ablation_bars.png"))

    # Print table
    print(f"\n{'='*60}")
    print(f"ABLATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Variant':<30} {'F1':>8} {'UDR':>8} {'FPR':>8}")
    print(f"{'-'*54}")
    for name, r in results.items():
        print(f"{name:<30} {r['F1']:8.4f} {r['UDR']:8.4f} {r['FPR']:8.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Smart Contract Vulnerability Detection Experiments")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test on 1000-sample subset")
    parser.add_argument("--skip-precompute", action="store_true",
                        help="Skip statistical feature precomputation")
    parser.add_argument("--ablation-only", action="store_true",
                        help="Only run ablation after existing training")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of seeds to run (default: 1)")
    parser.add_argument("--no-ablation", action="store_true",
                        help="Skip ablation study for faster results")
    args = parser.parse_args()

    config = Config()
    if args.quick:
        print("QUICK MODE: reduced dataset, fewer epochs")
        config.EPOCHS = 15
        config.BATCH_SIZE = 8

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    if args.ablation_only:
        print("ABLATION-ONLY MODE")
        config.EPOCHS = 10
        config.BATCH_SIZE = 8
        opc_vocab, param_vocab, atom_registry = build_all_vocabs(config)
        loaders = create_dataloaders(config, opc_vocab, param_vocab, atom_registry)
        if args.quick:
            loaders["train"].dataset.samples = loaders["train"].dataset.samples[:600]
            loaders["val"].dataset.samples = loaders["val"].dataset.samples[:150]
            loaders["test"].dataset.samples = loaders["test"].dataset.samples[:250]
        run_all_ablations(config, opc_vocab, param_vocab, atom_registry, loaders)
        return

    # Full experiment pipeline
    start_time = time.time()
    all_results = []

    for seed in config.SEEDS[:args.seeds]:
        result = run_single_experiment(config, seed=seed, quick=args.quick,
                                       skip_precompute=args.skip_precompute)
        all_results.append(result)
        # First seed also drives ablation (unless --no-ablation)
        if not args.no_ablation and seed == config.SEEDS[0]:
            opc_vocab, param_vocab, atom_registry = build_all_vocabs(config)
            loaders = create_dataloaders(config, opc_vocab, param_vocab, atom_registry)
            if args.quick:
                loaders["train"].dataset.samples = loaders["train"].dataset.samples[:600]
                loaders["val"].dataset.samples = loaders["val"].dataset.samples[:150]
                loaders["test"].dataset.samples = loaders["test"].dataset.samples[:250]
            run_all_ablations(config, opc_vocab, param_vocab, atom_registry, loaders)

    # Aggregate across seeds
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"AGGREGATED RESULTS ({len(all_results)} seeds)")
        print(f"{'='*60}")
        keys = ["macro_f1", "weighted_f1", "accuracy"]
        for key in keys:
            vals = [r["known"][key] for r in all_results]
            print(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        for key in ["UDR", "FPR", "F1_unknown", "ROC_AUC", "OSCA"]:
            vals = [r["unknown"][key] for r in all_results]
            print(f"  {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

        # Save aggregate
        known_keys = ["macro_f1", "weighted_f1", "accuracy"]
        unknown_keys = ["UDR", "FPR", "F1_unknown", "ROC_AUC", "OSCA"]
        aggregate = {}
        for k in known_keys:
            vals = [r["known"][k] for r in all_results]
            aggregate[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for k in unknown_keys:
            vals = [r["unknown"][k] for r in all_results]
            aggregate[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        with open(os.path.join(config.OUTPUT_DIR, "aggregate_summary.json"), 'w') as f:
            json.dump({k: {"mean": float(v["mean"]), "std": float(v["std"])}
                       for k, v in aggregate.items()}, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nAll experiments completed in {elapsed:.1f}s")
    print(f"Results saved to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
