#!/usr/bin/env python3
"""
Evaluation metrics for known-class classification and unknown-vulnerability detection.
Section 5.4 of the paper.
"""

import numpy as np
from collections import defaultdict


def compute_known_metrics(y_true, y_pred, num_classes, class_names=None):
    """
    Compute per-class and aggregated classification metrics.

    Returns dict with:
      per_class: list of {precision, recall, f1, support} per class
      macro_precision, macro_recall, macro_f1
      weighted_precision, weighted_recall, weighted_f1
      accuracy
      confusion_matrix: (K, K)
    """
    K = num_classes
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t >= 0 and p >= 0:
            cm[t, p] += 1

    per_class = []
    for k in range(K):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        support = cm[k, :].sum()
        prec = tp / (tp + fp + 1e-10)
        rec = tp / (tp + fn + 1e-10)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        name = class_names[k] if class_names else f"Class_{k}"
        per_class.append({"name": name, "precision": prec, "recall": rec,
                          "f1": f1, "support": int(support)})

    # Macro average
    macro_prec = np.mean([c["precision"] for c in per_class if c["support"] > 0])
    macro_rec = np.mean([c["recall"] for c in per_class if c["support"] > 0])
    macro_f1 = np.mean([c["f1"] for c in per_class if c["support"] > 0])

    # Weighted average
    total = cm.sum()
    weighted_prec = sum(c["precision"] * c["support"] for c in per_class) / (total + 1e-10)
    weighted_rec = sum(c["recall"] * c["support"] for c in per_class) / (total + 1e-10)
    weighted_f1 = sum(c["f1"] * c["support"] for c in per_class) / (total + 1e-10)

    accuracy = np.diag(cm).sum() / (total + 1e-10)

    return {
        "per_class": per_class,
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_prec),
        "weighted_recall": float(weighted_rec),
        "weighted_f1": float(weighted_f1),
        "accuracy": float(accuracy),
        "confusion_matrix": cm.tolist(),
    }


def compute_unknown_metrics(detection_results):
    """
    Compute unknown-vulnerability detection metrics.

    detection_results: list of dicts from detect_unknown()

    Returns:
      UDR (unknown detection rate), FPR (false positive rate),
      F1_unknown, ROC_AUC, AUPR, OSCA
    """
    y_true_unknown = []
    y_pred_unknown = []
    unknown_scores = []  # 1 - max_prob  (higher = more likely unknown)
    known_correct = []
    labels = []
    preds = []

    for r in detection_results:
        is_unk = r["is_unknown"]
        is_pred_unk = (r["decision"] == "unknown")
        y_true_unknown.append(1 if is_unk else 0)
        y_pred_unknown.append(1 if is_pred_unk else 0)
        unknown_scores.append(1.0 - r["max_prob"])
        if not is_unk:
            known_correct.append(r["pred_class"] == r["true_label"] and not is_pred_unk)
        labels.append(r["true_label"])
        preds.append(r["pred_class"])

    y_true_unknown = np.array(y_true_unknown)
    y_pred_unknown = np.array(y_pred_unknown)
    unknown_scores = np.array(unknown_scores)

    # UDR = recall of unknown class
    tp = ((y_true_unknown == 1) & (y_pred_unknown == 1)).sum()
    fn = ((y_true_unknown == 1) & (y_pred_unknown == 0)).sum()
    fp = ((y_true_unknown == 0) & (y_pred_unknown == 1)).sum()
    tn = ((y_true_unknown == 0) & (y_pred_unknown == 0)).sum()

    udr = tp / (tp + fn + 1e-10)  # recall
    fpr = fp / (fp + tn + 1e-10) if (fp + tn) > 0 else 0.0  # false positive rate
    prec_unk = tp / (tp + fp + 1e-10)
    f1_unknown = 2 * prec_unk * udr / (prec_unk + udr + 1e-10)

    # ROC-AUC
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        roc_auc = roc_auc_score(y_true_unknown, unknown_scores)
        aupr = average_precision_score(y_true_unknown, unknown_scores)
    except ImportError:
        roc_auc = 0.0
        aupr = 0.0

    # OSCA
    osca = (tp + sum(known_correct)) / len(detection_results) if detection_results else 0

    return {
        "UDR": float(udr),
        "FPR": float(fpr),
        "F1_unknown": float(f1_unknown),
        "ROC_AUC": float(roc_auc),
        "AUPR": float(aupr),
        "OSCA": float(osca),
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
    }


def print_known_metrics(metrics, title="Known Classification Metrics"):
    """Pretty-print known classification metrics."""
    print(f"\n  {title}")
    print(f"  {'-'*60}")
    print(f"  {'Class':<25} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Support':>7}")
    for c in metrics["per_class"]:
        print(f"  {c['name']:<25} {c['precision']:7.3f} {c['recall']:7.3f} "
              f"{c['f1']:7.3f} {c['support']:7d}")
    print(f"  {'-'*60}")
    print(f"  {'Macro avg':<25} {metrics['macro_precision']:7.3f} "
          f"{metrics['macro_recall']:7.3f} {metrics['macro_f1']:7.3f}")
    print(f"  {'Weighted avg':<25} {metrics['weighted_precision']:7.3f} "
          f"{metrics['weighted_recall']:7.3f} {metrics['weighted_f1']:7.3f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")


def print_unknown_metrics(metrics, title="Unknown Detection Metrics"):
    """Pretty-print unknown detection metrics."""
    print(f"\n  {title}")
    print(f"  {'-'*45}")
    print(f"  UDR (Unknown Detection Rate):  {metrics['UDR']:.4f}")
    print(f"  FPR (False Positive Rate):     {metrics['FPR']:.4f}")
    print(f"  F1@unknown:                    {metrics['F1_unknown']:.4f}")
    print(f"  ROC-AUC:                       {metrics['ROC_AUC']:.4f}")
    print(f"  AUPR:                          {metrics['AUPR']:.4f}")
    print(f"  OSCA:                          {metrics['OSCA']:.4f}")
