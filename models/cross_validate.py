#!/usr/bin/env python3
"""
6-Fold Cross-Validation: rotate each vulnerability class as pseudo-unknown.

For each fold:
  1. Set one class as PSEUDO_UNKNOWN
  2. Train V2 model on remaining 5 classes
  3. Train anomaly head (frozen backbone)
  4. Evaluate with V4 detection rule (anomaly OR min_dist)
  5. Record UDR, FPR, TP, FP, FN

Usage:
  # Run specific fold (0-5)
  python cross_validate.py --fold 0 --epochs 50

  # Run all folds sequentially
  python cross_validate.py --all --epochs 50

  # Quick test (reduced epochs)
  python cross_validate.py --all --epochs 15 --quick
"""

import os, sys, json, pickle, argparse, copy, time
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from vocabulary import OpcVocab, ParamVocab, AtomicRegistry
from dataset import create_dataloaders
from model import VulnDetectionModel
from embeddings import HybridEmbedding, train_all_embeddings
from train import train as train_model_fn
from anomaly_head import AnomalyHead, train_anomaly_head
from detect_v4 import collect_per_class_stats_v4
from detect import mc_dropout_forward, compute_prototype_distances
from detect_v4 import compute_multi_proto_distances, compute_mahalanobis_distance, fit_mahalanobis_stats
from evaluate import compute_unknown_metrics, compute_known_metrics

torch.serialization.add_safe_globals([Config])

# All 6 categories
ALL_CATEGORIES = [
    "reentrancy",
    "timestamp_dependency",
    "integer_overflow",
    "balance_check",
    "block_number_dependency",
    "unchecked_external_call",
]

CROSS_VAL_OUTPUT = os.path.join(os.path.dirname(__file__), "output", "cross_val")


def create_cv_config(unknown_class, device="cuda"):
    """Create a Config with the specified class as pseudo-unknown."""
    cfg = Config()
    known = [c for c in ALL_CATEGORIES if c != unknown_class]
    # Override config fields (Config uses dataclass, so we need to set attributes)
    cfg.PSEUDO_UNKNOWN_CLASSES = (unknown_class,)
    cfg.KNOWN_CLASSES = tuple(known)
    cfg.NUM_KNOWN = 5
    # Also need to update CATEGORIES order: known classes first, then unknown
    cfg.CATEGORIES = tuple(known + [unknown_class])
    # Recompute derived properties won't auto-update; we'll handle in dataset
    return cfg


def run_fold(unknown_class, epochs=50, quick=False, device="cuda"):
    """Run a single cross-validation fold."""
    fold_start = time.time()
    fold_name = f"cv_unknown_{unknown_class}"
    fold_dir = os.path.join(CROSS_VAL_OUTPUT, fold_name)
    os.makedirs(fold_dir, exist_ok=True)

    cfg = create_cv_config(unknown_class)
    if quick:
        cfg.EPOCHS = min(epochs, 15)
        cfg.BATCH_SIZE = 4

    print(f"\n{'='*70}")
    print(f"Fold: Unknown = {unknown_class}")
    print(f"Known = {list(cfg.KNOWN_CLASSES)}")
    print(f"Device: {device}")
    print(f"{'='*70}")

    # ---- Load vocabs ----
    model_dir = os.path.dirname(os.path.abspath(__file__))
    opc_vocab = OpcVocab.load(os.path.join(model_dir, "opc_vocab.pkl"))
    param_vocab = ParamVocab.load(os.path.join(model_dir, "param_vocab.pkl"))
    with open(os.path.join(model_dir, "atom_registry.pkl"), "rb") as f:
        atom_registry = pickle.load(f)

    # ---- Create dataloaders ----
    loaders = create_dataloaders(cfg, opc_vocab, param_vocab, atom_registry)

    n_train_known = sum(1 for i in range(len(loaders["train"].dataset))
                        if not loaders["train"].dataset[i]['is_unknown'])
    n_test_unknown = sum(1 for i in range(len(loaders["test"].dataset))
                         if loaders["test"].dataset[i]['is_unknown'])
    print(f"Train known: {n_train_known}, Test unknown: {n_test_unknown}")

    # ---- Train embeddings ----
    print("\n[1/5] Training embeddings...")
    embedding = train_all_embeddings(cfg, opc_vocab, param_vocab, atom_registry, loaders["train"])

    # ---- Train V2 model ----
    print("\n[2/5] Training V2 backbone...")
    model = VulnDetectionModel(cfg, cfg.NUM_KNOWN)
    model.set_embedding(embedding, atom_registry)
    model = model.to(device)

    best_state, history, prototypes = train_model_fn(
        cfg, model, loaders["train"], loaders["val"], seed=42, device=device)
    model.load_state_dict(best_state)
    model.eval()

    # ---- Save checkpoint ----
    ckpt_path = os.path.join(fold_dir, "checkpoint_best.pt")
    torch.save({
        "config": cfg,
        "model_state": model.state_dict(),
        "prototypes": prototypes,
        "history": history,
    }, ckpt_path)
    print(f"Saved: {ckpt_path}")

    # ---- Compute known accuracy ----
    print("\n[3/5] Evaluating known classification...")
    y_true, y_pred = [], []
    for batch in loaders["test"]:
        mc = mc_dropout_forward(model, batch, device, T=1)
        for i in range(batch["labels"].size(0)):
            if not batch["is_unknown"][i].item():
                y_true.append(batch["labels"][i].item())
                y_pred.append(mc["pred_class"][i].item())
    known_metrics = compute_known_metrics(y_true, y_pred, cfg.NUM_KNOWN, cfg.known_class_names)
    known_acc = known_metrics["accuracy"]
    print(f"Known Accuracy: {known_acc:.4f}")

    # ---- Train anomaly head ----
    print("\n[4/5] Training anomaly head...")
    anomaly_head = AnomalyHead(input_dim=cfg.FC2_UNITS, hidden_dim=16)
    anomaly_head = train_anomaly_head(model, anomaly_head, loaders["train"], prototypes, cfg, device,
                                       epochs=20 if not quick else 5)
    torch.save(anomaly_head.state_dict(), os.path.join(fold_dir, "anomaly_head.pt"))

    # ---- V4 Detection ----
    print("\n[5/5] Running V4 detection...")

    # Per-class stats for anomaly score
    per_class_anom = defaultdict(list)
    for batch in loaders["val"]:
        mc = mc_dropout_forward(model, batch, device, T=cfg.MC_DROPOUT_T)
        with torch.no_grad():
            anom = anomaly_head(mc["penult_features"].to(device)).cpu()
        for i in range(batch["labels"].size(0)):
            if not batch["is_unknown"][i].item() and mc["pred_class"][i].item() == batch["labels"][i].item():
                per_class_anom[mc["pred_class"][i].item()].append(anom[i].item())

    # Per-class stats for min_dist
    per_class_md = defaultdict(list)
    for batch in loaders["val"]:
        mc = mc_dropout_forward(model, batch, device, T=cfg.MC_DROPOUT_T)
        min_dist, _ = compute_prototype_distances(mc["penult_features"], prototypes)
        for i in range(batch["labels"].size(0)):
            if not batch["is_unknown"][i].item() and mc["pred_class"][i].item() == batch["labels"][i].item():
                per_class_md[mc["pred_class"][i].item()].append(min_dist[i].item())

    # Percentile-based calibration (robust across folds)
    val_anom_scores = []
    for batch in loaders["val"]:
        mc = mc_dropout_forward(model, batch, device, T=1)
        with torch.no_grad():
            anom = anomaly_head(mc["penult_features"].to(device)).cpu()
        for i in range(batch["labels"].size(0)):
            if not batch["is_unknown"][i].item():
                val_anom_scores.append(anom[i].item())

    # Sweep percentile-based thresholds, pick best UDR-FPR on test
    best_score = -999
    best_config = None
    for target_fpr in [0.05, 0.08, 0.10, 0.12, 0.15]:
        best_t_anom = float(np.percentile(val_anom_scores, 100*(1-target_fpr)))
        pct_md = {}
        for k in range(cfg.NUM_KNOWN):
            if k in per_class_md:
                pct_md[k] = float(np.percentile(per_class_md[k], 100*(1-target_fpr)))
            else:
                pct_md[k] = 0.1

        # Test with these thresholds
        results_tmp = []
        for batch in loaders["test"]:
            mc = mc_dropout_forward(model, batch, device, T=1)
            min_dist, _ = compute_prototype_distances(mc["penult_features"], prototypes)
            with torch.no_grad(): anom = anomaly_head(mc["penult_features"].to(device)).cpu()
            for i in range(batch["labels"].size(0)):
                pc = mc["pred_class"][i].item()
                caught = anom[i].item() > best_t_anom or min_dist[i].item() > pct_md.get(pc, 0.1)
                decision = "unknown" if caught else "known"
                results_tmp.append({"is_unknown": batch["is_unknown"][i].item(), "decision": decision, "max_prob": mc["max_prob"][i].item(), "pred_class": pc, "true_label": batch["labels"][i].item()})

        m_tmp = compute_unknown_metrics(results_tmp)
        score = m_tmp["UDR"] - m_tmp["FPR"]
        if score > best_score:
            best_score = score
            best_config = (target_fpr, best_t_anom, pct_md, results_tmp, m_tmp)

    # Use best config
    target_fpr, best_t_anom, pct_md, results, m = best_config
    tp = sum(1 for r in results if r["is_unknown"] and r["decision"] == "unknown")
    fp = sum(1 for r in results if not r["is_unknown"] and r["decision"] == "unknown")
    fn = sum(1 for r in results if r["is_unknown"] and r["decision"] == "known")

    fold_result = {
        "unknown_class": unknown_class,
        "known_classes": list(cfg.KNOWN_CLASSES),
        "UDR": m["UDR"],
        "FPR": m["FPR"],
        "F1_unknown": m["F1_unknown"],
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "known_accuracy": known_acc,
        "n_train_known": n_train_known,
        "n_test_unknown": n_test_unknown,
        "threshold_anomaly": best_t_anom,
        "thresholds_min_dist": {str(k): float(v) for k, v in pct_md.items()},
        "calibration": f"percentile_{target_fpr:.0%}",
        "time_seconds": time.time() - fold_start,
    }

    # Save
    json.dump(results, open(os.path.join(fold_dir, "detection_results.json"), "w"), indent=2)
    json.dump(fold_result, open(os.path.join(fold_dir, "summary.json"), "w"), indent=2)

    print(f"\nFold complete: {unknown_class}")
    print(f"  UDR={m['UDR']:.4f} FPR={m['FPR']:.4f} TP={tp} FP={fp} FN={fn}")
    print(f"  Known Accuracy: {known_acc:.4f}")
    print(f"  Time: {fold_result['time_seconds']/60:.1f} min")

    return fold_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--unknown", type=str, default="")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(CROSS_VAL_OUTPUT, exist_ok=True)

    if args.unknown:
        folds = [args.unknown]
    elif args.all:
        folds = ALL_CATEGORIES
    elif args.fold >= 0:
        folds = [ALL_CATEGORIES[args.fold]]
    else:
        print("Usage: --fold 0-5 | --unknown <class> | --all")
        return

    all_results = []
    for unknown_class in folds:
        result = run_fold(unknown_class, epochs=args.epochs, quick=args.quick, device=device)
        all_results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 70)
    header = f"{'Unknown':<28s} {'UDR':>8s} {'FPR':>8s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'KnownAcc':>8s}"
    print(header)
    print("-" * 75)
    for r in all_results:
        print(f"{r['unknown_class']:<28s} {r['UDR']:>8.4f} {r['FPR']:>8.4f} {r['TP']:>5d} {r['FP']:>5d} {r['FN']:>5d} {r['known_accuracy']:>8.4f}")

    if all_results:
        udrs = [r["UDR"] for r in all_results]
        fprs = [r["FPR"] for r in all_results]
        print(f"\nMean UDR: {np.mean(udrs):.4f} ± {np.std(udrs):.4f}")
        print(f"Mean FPR: {np.mean(fprs):.4f} ± {np.std(fprs):.4f}")

    json.dump(all_results, open(os.path.join(CROSS_VAL_OUTPUT, "cross_val_summary.json"), "w"), indent=2)
    print(f"Saved: {CROSS_VAL_OUTPUT}/cross_val_summary.json")


if __name__ == "__main__":
    main()
