#!/usr/bin/env python3
"""
Train anomaly head on frozen V2 backbone, then evaluate.
"""

import os, sys, json, pickle, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from vocabulary import OpcVocab, ParamVocab, AtomicRegistry
from dataset import create_dataloaders
from model import VulnDetectionModel
from embeddings import HybridEmbedding
from anomaly_head import AnomalyHead, train_anomaly_head
from detect_v2 import (
    detect_unknown_v2, calibrate_ood_threshold, collect_per_class_stats,
    compute_ood_score,
)
from evaluate import compute_unknown_metrics, compute_known_metrics

torch.serialization.add_safe_globals([Config])


def detect_v3(model, anomaly_head, dataloader, prototypes, config,
              per_class_stats, ood_threshold, device="cpu"):
    """
    V3 detection: V2 OOD score with anomaly head as 4th z-score dimension.
    No extra threshold — anomaly score is normalized per-class like other signals.
    """
    from detect import mc_dropout_forward, compute_prototype_distances
    from detect_v2 import compute_ood_score
    from tqdm import tqdm

    anomaly_head.eval()

    # Collect anomaly score stats per class from validation set (for normalization)
    # This is done in collect_per_class_stats_v3

    all_results = []
    for batch in tqdm(dataloader, desc="  Detecting v3"):
        mc = mc_dropout_forward(model, batch, device, T=config.MC_DROPOUT_T)
        min_dist, _ = compute_prototype_distances(mc["penult_features"], prototypes)

        with torch.no_grad():
            anomaly_scores = anomaly_head(mc["penult_features"].to(device))

        B = batch["labels"].size(0)
        for i in range(B):
            mp = mc["max_prob"][i].item()
            ent = mc["entropy"][i].item()
            md = min_dist[i].item()
            pc = mc["pred_class"][i].item()
            anom = anomaly_scores[i].item()

            ood = compute_ood_score(mp, ent, md, pc, per_class_stats,
                                     anomaly_score=anom, anom_stats=per_class_stats[pc].get("anom"))

            if ood <= ood_threshold:
                decision = "known"
            else:
                decision = "unknown"

            all_results.append({
                "true_label": batch["labels"][i].item(),
                "is_unknown": batch["is_unknown"][i].item(),
                "category": batch.get("category", [""])[i] if "category" in batch else "",
                "basename": batch["basename"][i] if "basename" in batch else "",
                "max_prob": mp, "entropy": ent, "min_dist": md,
                "pred_class": pc, "decision": decision,
                "ood_score": ood, "anomaly_score": anom,
            })

    return all_results


def collect_per_class_stats_v3(model, anomaly_head, dataloader, prototypes, config, device):
    """Extended per-class stats including anomaly scores."""
    from detect import mc_dropout_forward, compute_prototype_distances
    from collections import defaultdict
    import numpy as np

    print("  Collecting per-class stats (incl. anomaly scores)...")
    anomaly_head.eval()

    per_class = defaultdict(lambda: {"prob": [], "ent": [], "dist": [], "anom": []})

    for batch in dataloader:
        known_mask = ~batch["is_unknown"]
        if known_mask.sum() == 0:
            continue
        mc = mc_dropout_forward(model, batch, device, T=config.MC_DROPOUT_T)
        min_dist, _ = compute_prototype_distances(mc["penult_features"], prototypes)
        with torch.no_grad():
            anom = anomaly_head(mc["penult_features"].to(device)).cpu()

        for i in range(batch["labels"].size(0)):
            if not batch["is_unknown"][i].item():
                pred = mc["pred_class"][i].item()
                true = batch["labels"][i].item()
                if pred == true:  # only correctly classified
                    per_class[true]["prob"].append(mc["max_prob"][i].item())
                    per_class[true]["ent"].append(mc["entropy"][i].item())
                    per_class[true]["dist"].append(min_dist[i].item())
                    per_class[true]["anom"].append(anom[i].item())

    stats = {}
    for k in sorted(per_class.keys()):
        prob = np.array(per_class[k]["prob"])
        ent = np.array(per_class[k]["ent"])
        dist = np.array(per_class[k]["dist"])
        anom = np.array(per_class[k]["anom"])
        stats[k] = {
            "mean_prob": float(np.mean(prob)), "std_prob": float(np.std(prob)) + 1e-8,
            "mean_ent": float(np.mean(ent)), "std_ent": float(np.std(ent)) + 1e-8,
            "mean_dist": float(np.mean(dist)), "std_dist": float(np.std(dist)) + 1e-8,
            "anom": {"mean": float(np.mean(anom)), "std": float(np.std(anom)) + 1e-8},
            "n": len(prob),
        }
        print(f"    Class {k}: n={len(prob):>4d}  anom={stats[k]['anom']['mean']:.3f}±{stats[k]['anom']['std']:.3f}")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="output/seed_42_v2/checkpoint_best.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    # Load V2 checkpoint
    ckpt_path = os.path.join(os.path.dirname(__file__), args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg, model_state, saved_proto = ckpt["config"], ckpt["model_state"], ckpt["prototypes"]
    device = cfg.DEVICE

    if args.quick:
        cfg.EPOCHS = 15
        args.epochs = 5

    # Load vocabs
    model_dir = os.path.dirname(__file__)
    opc_vocab = OpcVocab.load(os.path.join(model_dir, "opc_vocab.pkl"))
    param_vocab = ParamVocab.load(os.path.join(model_dir, "param_vocab.pkl"))
    with open(os.path.join(model_dir, "atom_registry.pkl"), "rb") as f:
        atom_registry = pickle.load(f)

    loaders = create_dataloaders(cfg, opc_vocab, param_vocab, atom_registry)

    # Build model from V2 checkpoint
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

    print(f"Loaded V2 model from: {args.checkpoint}")
    print(f"Device: {device}")

    # ---- Train anomaly head ----
    print("\n[1/3] Training anomaly head...")
    anomaly_head = AnomalyHead(input_dim=cfg.FC2_UNITS, hidden_dim=16)
    train_anomaly_head(model, anomaly_head, loaders["train"], prototypes, cfg, device,
                       epochs=args.epochs)

    # ---- Run detection ----
    print("\n[2/3] Running detection...")

    # V2 baseline stats
    pcs_v2 = collect_per_class_stats(model, loaders["val"], prototypes, cfg, device)

    # V3 stats (includes anomaly scores per class)
    pcs_v3 = collect_per_class_stats_v3(model, anomaly_head, loaders["val"], prototypes, cfg, device)

    # Calibrate OOD threshold (V2 method — same for both, but V3 uses different per-class stats)
    ood_thresh_v2 = calibrate_ood_threshold(model, loaders["val"], prototypes, cfg, pcs_v2, device)
    ood_thresh_v3 = calibrate_ood_threshold(model, loaders["val"], prototypes, cfg, pcs_v3, device)

    # V2 baseline
    r2 = detect_unknown_v2(model, loaders["test"], prototypes, cfg, pcs_v2,
                            ood_threshold=ood_thresh_v2, device=device)
    m2 = compute_unknown_metrics(r2)

    # V3 (V2 + anomaly head as 4th z-score dimension)
    r3 = detect_v3(model, anomaly_head, loaders["test"], prototypes, cfg, pcs_v3,
                    ood_threshold=ood_thresh_v3, device=device)
    m3 = compute_unknown_metrics(r3)

    # ---- Compare ----
    print("\n[3/3] Results")
    print("=" * 60)
    print(f"{'Metric':<12s} {'V2':>10s} {'V3 (+Anomaly)':>16s}")
    print("-" * 40)
    for key in ["UDR", "FPR", "F1_unknown"]:
        print(f"{key:<12s} {m2[key]:>10.4f} {m3[key]:>16.4f}")
    tp2 = sum(1 for r in r2 if r["is_unknown"] and r["decision"] == "unknown")
    tp3 = sum(1 for r in r3 if r["is_unknown"] and r["decision"] == "unknown")
    fp2 = sum(1 for r in r2 if not r["is_unknown"] and r["decision"] == "unknown")
    fp3 = sum(1 for r in r3 if not r["is_unknown"] and r["decision"] == "unknown")
    print(f"{'TP':<12s} {tp2:>10d} {tp3:>16d}")
    print(f"{'FP':<12s} {fp2:>10d} {fp3:>16d}")

    print(f"\n{'Category':<28s} {'V2 FPR':>8s} {'V3 FPR':>8s}")
    print("-" * 46)
    for cat in cfg.KNOWN_CLASSES:
        items2 = [r for r in r2 if r["category"] == cat and not r["is_unknown"]]
        items3 = [r for r in r3 if r["category"] == cat and not r["is_unknown"]]
        if items2:
            fpr2 = sum(1 for r in items2 if r["decision"] == "unknown") / len(items2)
            fpr3 = sum(1 for r in items3 if r["decision"] == "unknown") / len(items3)
            print(f"  {cat:<26s} {fpr2:>8.4f} {fpr3:>8.4f}")

    # Save
    out_dir = os.path.dirname(ckpt_path)
    torch.save(anomaly_head.state_dict(), os.path.join(out_dir, "anomaly_head.pt"))
    json.dump(r3, open(os.path.join(out_dir, "detection_results_v3.json"), "w"), indent=2)
    print(f"\nSaved: {out_dir}/anomaly_head.pt, detection_results_v3.json")


if __name__ == "__main__":
    main()
