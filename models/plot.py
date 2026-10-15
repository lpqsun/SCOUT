#!/usr/bin/env python3
"""
Visualization utilities for vulnerability detection experiments.
Section 5.8: Suggested figure list.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def set_style():
    """Set consistent plot style."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
    })


def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    """Heatmap of known-class confusion matrix."""
    set_style()
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=10, color='white' if cm[i, j] > cm.max()/2 else 'black')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Count')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_roc_curves(fpr_tpr_list, method_names, save_path,
                    title="ROC Curves — Unknown Detection"):
    """Multi-line ROC curve comparison."""
    set_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    for (fpr, tpr, auc_val), name in zip(fpr_tpr_list, method_names):
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc='lower right')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_ablation_bars(results_dict, save_path,
                       title="Ablation Study"):
    """Grouped bar chart for ablation variants."""
    set_style()
    variants = list(results_dict.keys())
    metrics = ['F1', 'UDR', 'FPR']
    x = np.arange(len(variants))
    width = 0.25
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, metric in enumerate(metrics):
        vals = [results_dict[v].get(metric, 0) for v in variants]
        bars = ax.bar(x + i * width, vals, width, label=metric, color=colors[i])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels([v.replace('_', '\n') for v in variants], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1.15)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_tsne(features, labels, class_names, save_path,
              title="t-SNE — Feature Space (32-dim penultimate)"):
    """t-SNE visualization of penultimate features."""
    from sklearn.manifold import TSNE
    set_style()
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    markers = ['o', 's', '^', 'v', 'D', 'x']
    for i, lbl in enumerate(unique_labels):
        mask = np.array(labels) == lbl
        name = class_names[lbl] if lbl < len(class_names) else f"Unknown_{lbl}"
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                   c=[colors[i]], marker=markers[i % len(markers)],
                   label=name, alpha=0.6, s=10)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.legend(markerscale=2, fontsize=8)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_perf_vs_length(length_bins, f1_per_bin, save_path,
                        title="F1 vs Sequence Length"):
    """Line plot of performance against sequence length buckets."""
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    bin_centers = [(a + b) / 2 for a, b in length_bins]
    ax.plot(bin_centers, f1_per_bin, 'o-', color='#2ecc71', linewidth=2, markersize=8)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_attention_heatmap(attn_weights, opcode_seq, save_path, max_len=100,
                           title="Attention Weight Heatmap"):
    """Single-sample attention weight heatmap."""
    set_style()
    attn = attn_weights[:max_len]
    opcodes = opcode_seq[:max_len] if len(opcode_seq) > max_len else opcode_seq
    L = len(attn)

    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(attn.reshape(1, -1), aspect='auto', cmap='hot')
    ax.set_yticks([])
    # Show every 5th opcode label
    tick_positions = range(0, L, 5)
    tick_labels = [opcodes[i] if i < len(opcodes) else "" for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_xlabel("Sequence Position")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Attention weight')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_training_history(history, save_path, title="Training Curves"):
    """Plot training and validation loss/accuracy curves."""
    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], 'b-', label='Train Loss')
    ax1.plot(epochs, history["val_loss"], 'r-', label='Val Loss')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], 'b-', label='Train Acc')
    ax2.plot(epochs, history["val_acc"], 'r-', label='Val Acc')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(title)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ==============================================================================
# Additional plots for paper experiments (Sections 5.4, 5.6.3, 5.6.4)
# ==============================================================================

def plot_pr_curve(y_true, y_scores, save_path,
                  title="Precision-Recall Curve — Unknown Detection"):
    """
    Precision-Recall curve for unknown vulnerability detection.
    Section 5.4: AUPR metric, especially useful for class imbalance.
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    set_style()

    prec, rec, _ = precision_recall_curve(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, linewidth=2, color='#2ecc71',
            label=f'AUPR = {aupr:.4f}')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return aupr


def plot_multi_roc_curves(methods_dict, save_path,
                          title="ROC Curves — Unknown Detection Comparison"):
    """
    Multiple ROC curves on one plot for baseline comparison.
    Section 5.3 / Figure 2.

    Args:
        methods_dict: {method_name: {"y_true": [...], "y_scores": [...]}}
    """
    from sklearn.metrics import roc_curve, auc
    set_style()

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    markers = ['o', 's', '^', 'D', 'v', 'x']

    for i, (name, data) in enumerate(methods_dict.items()):
        if not data.get("y_true") or not data.get("y_scores"):
            continue
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_scores"])
        roc_auc = auc(fpr, tpr)
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, linewidth=2, color=color, label=f'{name} (AUC={roc_auc:.3f})')

        # Mark operating point (FPR, TPR) if available
        if "fpr_point" in data and "tpr_point" in data:
            ax.scatter([data["fpr_point"]], [data["tpr_point"]],
                      color=color, s=80, marker=markers[i % len(markers)],
                      zorder=5, edgecolors='white', linewidths=1)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.2)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_baseline_comparison(baseline_metrics, save_path,
                             title="Baseline Comparison — Unknown Detection"):
    """
    Grouped bar chart comparing multiple methods across metrics.
    Section 5.3 / Table 3 visualization.

    Args:
        baseline_metrics: {method_name: {"UDR": ..., "FPR": ..., "F1_unknown": ..., ...}}
    """
    set_style()
    methods = list(baseline_metrics.keys())
    metric_keys = ["UDR", "F1_unknown", "ROC_AUC", "AUPR"]
    metric_labels = ["UDR", "F1@Unknown", "ROC-AUC", "AUPR"]

    x = np.arange(len(methods))
    width = 0.2
    colors = ['#2ecc71', '#3498db', '#f39c12', '#9b59b6']

    fig, ax = plt.subplots(figsize=(max(10, len(methods) * 1.8), 6))
    for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        vals = [baseline_metrics[m].get(key, 0) for m in methods]
        bars = ax.bar(x + i * width, vals, width, label=label, color=colors[i])
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', fontsize=7, rotation=90)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([m.replace(' ', '\n') for m in methods], fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.grid(axis='y', alpha=0.2)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_per_unknown_type(per_type_metrics, save_path,
                          title="Robustness — Detection Rate by Unknown Type"):
    """
    Bar chart showing unknown detection rate broken down by held-out class.
    Section 5.6.4: robustness against different unknown variants.

    Args:
        per_type_metrics: {unknown_class_name: {"UDR": ..., "count": ...}, ...}
    """
    set_style()
    types = list(per_type_metrics.keys())
    udr_vals = [per_type_metrics[t].get("UDR", 0) for t in types]
    counts = [per_type_metrics[t].get("count", 0) for t in types]
    colors = plt.cm.Set2(np.linspace(0, 1, len(types)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # UDR per type
    bars = ax1.bar(range(len(types)), udr_vals, color=colors, edgecolor='white')
    for bar, val in zip(bars, udr_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', fontsize=10)
    ax1.set_xticks(range(len(types)))
    ax1.set_xticklabels([t.replace('_', '\n') for t in types], fontsize=9)
    ax1.set_ylabel("Unknown Detection Rate (UDR)")
    ax1.set_title("Detection Rate by Unknown Type")
    ax1.set_ylim(0, 1.15)
    ax1.grid(axis='y', alpha=0.2)

    # Sample count per type
    bars2 = ax2.bar(range(len(types)), counts, color=colors, edgecolor='white')
    for bar, val in zip(bars2, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(val), ha='center', fontsize=10)
    ax2.set_xticks(range(len(types)))
    ax2.set_xticklabels([t.replace('_', '\n') for t in types], fontsize=9)
    ax2.set_ylabel("Sample Count")
    ax2.set_title("Samples per Unknown Type")
    ax2.grid(axis='y', alpha=0.2)

    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_robustness_length(all_results, save_path,
                           title="Robustness — Performance by Sequence Length"):
    """
    F1 and UDR across sequence length buckets.
    Section 5.6.4: robustness analysis.

    Args:
        all_results: list of result dicts from detect_unknown(),
                     each must have 'length' or derived from basename
    """
    set_style()
    bins = [(0, 500), (500, 1000), (1000, 2000), (2000, 3000), (3000, 5000), (5000, 99999)]
    bin_labels = [f"{lo}-{hi}" for lo, hi in bins]
    bin_centers = [(lo + min(hi, 10000)) / 2 for lo, hi in bins]

    # Compute UDR per bin
    udr_per_bin = []
    f1_per_bin = []
    for lo, hi in bins:
        bin_results = [r for r in all_results
                       if lo <= r.get("length", 0) < hi]
        if not bin_results:
            udr_per_bin.append(0)
            f1_per_bin.append(0)
            continue

        # UDR
        tp = sum(1 for r in bin_results
                if r["is_unknown"] and r["decision"] == "unknown")
        fn = sum(1 for r in bin_results
                if r["is_unknown"] and r["decision"] != "unknown")
        udr_per_bin.append(tp / (tp + fn + 1e-10))

        # Known F1 (approximate)
        known_correct = sum(1 for r in bin_results
                           if not r["is_unknown"]
                           and r.get("pred_class") == r.get("true_label"))
        known_total = sum(1 for r in bin_results if not r["is_unknown"])
        f1_per_bin.append(known_correct / (known_total + 1e-10))

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(bins))
    width = 0.35

    bars1 = ax.bar(x - width / 2, udr_per_bin, width, label='UDR',
                   color='#3498db', edgecolor='white')
    bars2 = ax.bar(x + width / 2, f1_per_bin, width, label='Known F1',
                   color='#2ecc71', edgecolor='white')

    for bar, val in zip(bars1, udr_per_bin):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', fontsize=8, rotation=90)
    for bar, val in zip(bars2, f1_per_bin):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', fontsize=8, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=9)
    ax.set_xlabel("Sequence Length Range")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1.2)
    ax.grid(axis='y', alpha=0.2)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_decision_distribution(detection_results, save_path,
                               title="Decision Distribution — Known vs Unknown Samples"):
    """
    Stacked bar chart showing decision distribution for known vs unknown samples.
    Useful for understanding the dual-threshold behavior.
    """
    set_style()
    known_decisions = {"known": 0, "unknown": 0, "ambiguous": 0}
    unknown_decisions = {"known": 0, "unknown": 0, "ambiguous": 0}

    for r in detection_results:
        dec = r.get("decision", r.get("stage2_decision", "ambiguous"))
        if r["is_unknown"]:
            unknown_decisions[dec] = unknown_decisions.get(dec, 0) + 1
        else:
            known_decisions[dec] = known_decisions.get(dec, 0) + 1

    categories = ["Known (test)", "Unknown (held-out)"]
    known_vals = [known_decisions["known"], unknown_decisions["known"]]
    unknown_vals = [known_decisions["unknown"], unknown_decisions["unknown"]]
    ambiguous_vals = [known_decisions["ambiguous"], unknown_decisions["ambiguous"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(categories))
    width = 0.6

    p1 = ax.bar(x, known_vals, width, label='Predicted Known', color='#2ecc71')
    p2 = ax.bar(x, unknown_vals, width, bottom=known_vals,
                label='Predicted Unknown', color='#e74c3c')
    p3 = ax.bar(x, ambiguous_vals, width,
                bottom=[a + b for a, b in zip(known_vals, unknown_vals)],
                label='Ambiguous', color='#f39c12')

    for i, (k, u, a) in enumerate(zip(known_vals, unknown_vals, ambiguous_vals)):
        total = k + u + a
        if total > 0:
            ax.text(i, total + total * 0.02, f'n={total}', ha='center', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Number of Samples")
    ax.set_title(title)
    ax.legend(loc='upper right')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")
