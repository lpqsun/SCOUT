#!/usr/bin/env python3
"""
Smart Contract Vulnerability Detection Models.

Components:
  - config.py          : Hyperparameters
  - vocabulary.py      : Token vocabularies
  - precompute.py      : Statistical feature precomputation
  - dataset.py         : PyTorch Dataset & DataLoader
  - embeddings.py      : CBOW embedding training
  - model.py           : CNN-BiLSTM-Attention-GCN model
  - train.py           : Training loop
  - stage1_svdd.py     : Stage 1 — Deep SVDD one-class anomaly detection
  - detect.py          : Stage 2 — Unknown vulnerability detection (MC Dropout + dual-threshold)
  - baselines.py       : Baseline methods (OpenMax, Energy-OOD, Mahalanobis, Softmax)
  - evaluate.py        : Evaluation metrics
  - plot.py            : Visualization (confusion matrix, ROC, PR, t-SNE, attention, etc.)
  - sensitivity.py     : Hyperparameter sensitivity analysis
  - stats_analysis.py  : Dataset statistics & computational efficiency
  - run_experiments.py : Full experiment runner (two-stage pipeline)
"""

from .config import Config
from .vocabulary import (
    build_all_vocabs, OpcVocab, ParamVocab, AtomicRegistry,
    ATOMIC_PARAM_TYPES,
)
from .dataset import VulnTraceDataset, create_dataloaders
from .embeddings import train_cbow, CBOWModel, HybridEmbedding, train_all_embeddings
from .model import VulnDetectionModel, SparseGCNLayer, build_model_variant
from .train import train
from .stage1_svdd import DeepSVDD, SVDDEncoder, TwoStageDetector, evaluate_stage1
from .detect import detect_unknown, calibrate_thresholds, mc_dropout_forward
from .evaluate import (
    compute_known_metrics, compute_unknown_metrics,
    print_known_metrics, print_unknown_metrics,
)
from .baselines import (
    baseline_softmax_threshold, baseline_openmax, baseline_energy_ood,
    baseline_mahalanobis, run_baseline_comparison,
    BASELINE_METHODS, BASELINE_LABELS,
)
from .plot import (
    plot_confusion_matrix, plot_roc_curves, plot_pr_curve,
    plot_multi_roc_curves, plot_baseline_comparison, plot_ablation_bars,
    plot_tsne, plot_perf_vs_length, plot_attention_heatmap,
    plot_training_history, plot_per_unknown_type,
    plot_robustness_length, plot_decision_distribution,
)
from .sensitivity import (
    run_sensitivity_analysis, grid_search_tau, mc_dropout_sensitivity,
    lmax_sensitivity,
)
from .stats_analysis import (
    run_stats_and_efficiency, compute_dataset_statistics,
    measure_efficiency, print_dataset_table, print_efficiency_table,
)

__all__ = [
    # Config
    "Config",
    # Vocabulary
    "build_all_vocabs", "OpcVocab", "ParamVocab", "AtomicRegistry",
    "ATOMIC_PARAM_TYPES",
    # Data
    "VulnTraceDataset", "create_dataloaders",
    # Embeddings
    "train_cbow", "CBOWModel", "HybridEmbedding", "train_all_embeddings",
    # Model
    "VulnDetectionModel", "SparseGCNLayer", "build_model_variant",
    # Training
    "train",
    # Stage 1: Anomaly Detection
    "DeepSVDD", "SVDDEncoder", "TwoStageDetector", "evaluate_stage1",
    # Stage 2: Unknown Detection
    "detect_unknown", "calibrate_thresholds", "mc_dropout_forward",
    # Evaluation
    "compute_known_metrics", "compute_unknown_metrics",
    "print_known_metrics", "print_unknown_metrics",
    # Baselines
    "baseline_softmax_threshold", "baseline_openmax", "baseline_energy_ood",
    "baseline_mahalanobis", "run_baseline_comparison",
    "BASELINE_METHODS", "BASELINE_LABELS",
    # Plotting
    "plot_confusion_matrix", "plot_roc_curves", "plot_pr_curve",
    "plot_multi_roc_curves", "plot_baseline_comparison", "plot_ablation_bars",
    "plot_tsne", "plot_perf_vs_length", "plot_attention_heatmap",
    "plot_training_history", "plot_per_unknown_type",
    "plot_robustness_length", "plot_decision_distribution",
    # Sensitivity
    "run_sensitivity_analysis", "grid_search_tau",
    "mc_dropout_sensitivity", "lmax_sensitivity",
    # Stats & Efficiency
    "run_stats_and_efficiency", "compute_dataset_statistics",
    "measure_efficiency", "print_dataset_table", "print_efficiency_table",
]
