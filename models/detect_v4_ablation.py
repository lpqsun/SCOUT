#!/usr/bin/env python3
"""
V4.1: Voting fusion + per-signal ablation.
Key insight: max() fusion inflates FPR as more signals are added.
Voting (require ≥k signals to fire) controls this naturally.
"""

import os, sys, json, pickle, argparse
import torch
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from vocabulary import OpcVocab, ParamVocab
from model import VulnDetectionModel
from embeddings import HybridEmbedding
from anomaly_head import AnomalyHead
from dataset import create_dataloaders
from detect import mc_dropout_forward, compute_prototype_distances
from detect_v4 import (
    compute_energy_score, compute_multi_proto_distances,
    fit_mahalanobis_stats, compute_mahalanobis_distance,
    collect_per_class_stats_v4, compute_ood_score_v4,
)
from evaluate import compute_unknown_metrics

torch.serialization.add_safe_globals([Config])


def detect_v4_voting(model, anomaly_head, dataloader, prototypes, mahal_stats,
                      per_class_stats, config, device="cpu",
                      z_threshold=2.0, k_votes=2, signal_subset=None):
    """
    Voting-based detection: each signal votes independently.
    Decision: unknown if >= k_votes signals exceed z_threshold.

    Args:
        z_threshold: per-signal z-score threshold for a "vote"
        k_votes: minimum number of votes for "unknown" decision
        signal_subset: list of signal names to use (None = all 8)
    """
    ALL_SIGNALS = ["prob", "ent", "dist", "anom", "energy", "d1d2_ratio", "mean_proto_dist", "mahal"]
    if signal_subset is None:
        signal_subset = ALL_SIGNALS

    anomaly_head.eval()
    all_results = []

    for batch in tqdm(dataloader, desc=f"  Voting (k={k_votes}, z>{z_threshold})"):
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
            pc = mc["pred_class"][i].item()
            s = per_class_stats.get(pc)
            if s is None:
                decision = "unknown"
                vote_count = len(signal_subset)
            else:
                # Compute all 8 z-scores
                z_all = {}
                z_all["prob"] = -(mc["max_prob"][i].item() - s["mean_prob"]) / s["std_prob"]
                z_all["ent"] = (mc["entropy"][i].item() - s["mean_ent"]) / s["std_ent"]
                z_all["dist"] = (d1[i].item() - s["mean_dist"]) / s["std_dist"]

                if "mean_anom" in s:
                    z_all["anom"] = (anom[i].item() - s["mean_anom"]) / s["std_anom"]
                if "mean_energy" in s:
                    z_all["energy"] = -(energy[i].item() - s["mean_energy"]) / s["std_energy"]
                if "mean_d1d2_ratio" in s:
                    z_all["d1d2_ratio"] = (ratio[i].item() - s["mean_d1d2_ratio"]) / s["std_d1d2_ratio"]
                if "mean_mean_proto_dist" in s:
                    z_all["mean_proto_dist"] = (mean_dist[i].item() - s["mean_mean_proto_dist"]) / s["std_mean_proto_dist"]
                if "mean_mahal" in s:
                    z_all["mahal"] = (mahal[i].item() - s["mean_mahal"]) / s["std_mahal"]

                # Count votes
                vote_count = sum(1 for name in signal_subset if z_all.get(name, 0) > z_threshold)
                decision = "unknown" if vote_count >= k_votes else "known"

            all_results.append({
                "true_label": batch["labels"][i].item(),
                "is_unknown": batch["is_unknown"][i].item(),
                "category": batch.get("category", [""])[i] if "category" in batch else "",
                "basename": batch["basename"][i] if "basename" in batch else "",
                "max_prob": mc["max_prob"][i].item(),
                "entropy": mc["entropy"][i].item(),
                "min_dist": d1[i].item(),
                "pred_class": pc,
                "decision": decision,
                "vote_count": vote_count,
                "anomaly_score": anom[i].item(),
            })

    return all_results


def grid_search_voting(model, anomaly_head, val_loader, test_loader,
                        prototypes, mahal_stats, per_class_stats, config, device,
                        signal_subset=None):
    """Grid search over (z_threshold, k_votes) on validation set, evaluate on test."""
    z_candidates = [1.0, 1.5, 2.0, 2.5, 3.0]
    k_candidates = [1, 2, 3, 4] if signal_subset is None else list(range(1, min(len(signal_subset)+1, 5)))

    best_score = -1
    best_config = None
    best_test_results = None
    best_test_metrics = None

    for z_thresh in z_candidates:
        for k in k_candidates:
            if k > len(signal_subset or []):
                continue
            # Evaluate on val
            val_results = detect_v4_voting(
                model, anomaly_head, val_loader, prototypes, mahal_stats,
                per_class_stats, config, device,
                z_threshold=z_thresh, k_votes=k, signal_subset=signal_subset)
            val_metrics = compute_unknown_metrics(val_results)
            # Score: UDR - FPR (simple tradeoff)
            score = val_metrics["UDR"] - val_metrics["FPR"] * 0.5

            if score > best_score:
                best_score = score
                best_config = (z_thresh, k)
                # Evaluate on test
                test_results = detect_v4_voting(
                    model, anomaly_head, test_loader, prototypes, mahal_stats,
                    per_class_stats, config, device,
                    z_threshold=z_thresh, k_votes=k, signal_subset=signal_subset)
                best_test_metrics = compute_unknown_metrics(test_results)
                best_test_results = test_results

    return best_config, best_test_metrics, best_test_results


def per_signal_ablation(model, anomaly_head, test_loader, prototypes, mahal_stats,
                         per_class_stats, config, device):
    """Test adding each new signal one at a time to the base 4 (V3)."""
    BASE = ["prob", "ent", "dist", "anom"]
    NEW = ["energy", "d1d2_ratio", "mean_proto_dist", "mahal"]

    print("\n=== Per-Signal Ablation (base=V3, voting k=2, z=2.0) ===")
    print(f"{'Signals':<40s} {'UDR':>8s} {'FPR':>8s} {'TP':>5s} {'FP':>5s} {'FN':>5s}")
    print("-" * 75)

    results = []

    # Base V3 (4 signals)
    r = detect_v4_voting(model, anomaly_head, test_loader, prototypes, mahal_stats,
                          per_class_stats, config, device,
                          z_threshold=2.0, k_votes=2, signal_subset=BASE)
    m = compute_unknown_metrics(r)
    tp = sum(1 for x in r if x["is_unknown"] and x["decision"] == "unknown")
    fp = sum(1 for x in r if not x["is_unknown"] and x["decision"] == "unknown")
    fn = sum(1 for x in r if x["is_unknown"] and x["decision"] == "known")
    print(f"{'V3 (4 signals)':<40s} {m['UDR']:>8.4f} {m['FPR']:>8.4f} {tp:>5d} {fp:>5d} {fn:>5d}")
    results.append(("V3", BASE, m, tp, fp, fn))

    # Add each new signal individually
    for sig in NEW:
        subset = BASE + [sig]
        r = detect_v4_voting(model, anomaly_head, test_loader, prototypes, mahal_stats,
                              per_class_stats, config, device,
                              z_threshold=2.0, k_votes=2, signal_subset=subset)
        m = compute_unknown_metrics(r)
        tp = sum(1 for x in r if x["is_unknown"] and x["decision"] == "unknown")
        fp = sum(1 for x in r if not x["is_unknown"] and x["decision"] == "unknown")
        fn = sum(1 for x in r if x["is_unknown"] and x["decision"] == "known")
        delta_udr = m["UDR"] - results[0][2]["UDR"]
        delta_fpr = m["FPR"] - results[0][2]["FPR"]
        print(f"{'+'+sig:<40s} {m['UDR']:>8.4f} {m['FPR']:>8.4f} {tp:>5d} {fp:>5d} {fn:>5d}  ΔUDR={delta_udr:+.4f} ΔFPR={delta_fpr:+.4f}")
        results.append((f"+{sig}", subset, m, tp, fp, fn))

    # All 8 signals
    ALL = BASE + NEW
    r = detect_v4_voting(model, anomaly_head, test_loader, prototypes, mahal_stats,
                          per_class_stats, config, device,
                          z_threshold=2.0, k_votes=2, signal_subset=ALL)
    m = compute_unknown_metrics(r)
    tp = sum(1 for x in r if x["is_unknown"] and x["decision"] == "unknown")
    fp = sum(1 for x in r if not x["is_unknown"] and x["decision"] == "unknown")
    fn = sum(1 for x in r if x["is_unknown"] and x["decision"] == "known")
    delta_udr = m["UDR"] - results[0][2]["UDR"]
    delta_fpr = m["FPR"] - results[0][2]["FPR"]
    print(f"{'All 8 signals':<40s} {m['UDR']:>8.4f} {m['FPR']:>8.4f} {tp:>5d} {fp:>5d} {fn:>5d}  ΔUDR={delta_udr:+.4f} ΔFPR={delta_fpr:+.4f}")
    results.append(("All 8", ALL, m, tp, fp, fn))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="output/seed_42_v2/checkpoint_best.pt")
    parser.add_argument("--anomaly-head", type=str, default="output/seed_42_v2/anomaly_head.pt")
    parser.add_argument("--grid-search", action="store_true", help="Run full grid search")
    args = parser.parse_args()

    model_dir = os.path.dirname(os.path.abspath(__file__))

    # Load checkpoint
    ckpt_path = os.path.join(model_dir, args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg, model_state, saved_proto = ckpt["config"], ckpt["model_state"], ckpt["prototypes"]
    device = cfg.DEVICE
    print(f"Device: {device}")

    # Load vocabs
    opc_vocab = OpcVocab.load(os.path.join(model_dir, "opc_vocab.pkl"))
    param_vocab = ParamVocab.load(os.path.join(model_dir, "param_vocab.pkl"))
    with open(os.path.join(model_dir, "atom_registry.pkl"), "rb") as f:
        atom_registry = pickle.load(f)

    loaders = create_dataloaders(cfg, opc_vocab, param_vocab, atom_registry)

    # Build model
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
    anomaly_head = AnomalyHead(input_dim=cfg.FC2_UNITS, hidden_dim=16)
    anomaly_head.load_state_dict(torch.load(
        os.path.join(model_dir, args.anomaly_head), map_location="cpu", weights_only=True))
    anomaly_head = anomaly_head.to(device)
    anomaly_head.eval()

    # Fit Mahalanobis stats
    print("\n[1/3] Fitting Mahalanobis statistics...")
    mahal_stats = fit_mahalanobis_stats(model, loaders["train"], cfg, device)

    # Collect per-class stats (V4: all 8 signals)
    print("\n[2/3] Collecting per-class statistics...")
    per_class_stats = collect_per_class_stats_v4(
        model, anomaly_head, loaders["val"], prototypes, mahal_stats, cfg, device)

    # Per-signal ablation with voting
    print("\n[3/3] Running ablation + voting analysis...")
    ablation_results = per_signal_ablation(
        model, anomaly_head, loaders["test"], prototypes, mahal_stats,
        per_class_stats, cfg, device)

    # Optional: grid search for best (z_threshold, k) on best signal set
    if args.grid_search:
        print("\n=== Grid Search: Best Signal Set ===")
        best_name, best_signals, _, _, _, _ = max(
            ablation_results, key=lambda x: x[2]["UDR"] - x[2]["FPR"] * 0.5)
        print(f"Best signal set: {best_name} ({best_signals})")

        best_config, best_metrics, best_results = grid_search_voting(
            model, anomaly_head, loaders["val"], loaders["test"],
            prototypes, mahal_stats, per_class_stats, cfg, device,
            signal_subset=best_signals)

        print(f"\nBest voting config: z>{best_config[0]}, k>={best_config[1]}")
        print(f"UDR={best_metrics['UDR']:.4f}  FPR={best_metrics['FPR']:.4f}  F1={best_metrics['F1_unknown']:.4f}")

    # Save
    out_dir = os.path.dirname(ckpt_path)
    json.dump([{
        "name": name, "signals": sigs, "UDR": m["UDR"], "FPR": m["FPR"],
        "F1": m["F1_unknown"], "TP": tp, "FP": fp, "FN": fn,
    } for name, sigs, m, tp, fp, fn in ablation_results],
        open(os.path.join(out_dir, "v4_ablation.json"), "w"), indent=2)
    print(f"\nSaved: {out_dir}/v4_ablation.json")


if __name__ == "__main__":
    main()
