#!/usr/bin/env python3
"""
Central configuration for the Smart Contract Vulnerability Detection system.
All hyperparameters from the paper, optimized for CPU training.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    """Central hyperparameter configuration."""

    # ==========================================================================
    # Device
    # ==========================================================================
    @property
    def DEVICE(self) -> str:
        """Auto-detect GPU if available, otherwise CPU."""
        if torch.cuda.is_available():
            return "cuda"
        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def DEVICE_NAME(self) -> str:
        if torch.cuda.is_available():
            return f"CUDA: {torch.cuda.get_device_name(0)}"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "Apple MPS"
        return "CPU"

    # ==========================================================================
    # Data Paths
    # ==========================================================================
    DATA_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PARAM_DIR_SUFFIX: str = "_ParamFeatures"

    # Six vulnerability categories in the dataset
    CATEGORIES: Tuple[str, ...] = (
        "reentrancy",
        "timestamp_dependency",
        "balance_check",
        "block_number_dependency",
        "integer_overflow",
        "unchecked_external_call",
    )

    # Class display names keyed by category directory name (not label index).
    # Use known_class_names / all_class_names properties for label-indexed access.
    CLASS_NAMES: dict = field(default_factory=lambda: {
        "reentrancy":               "Reentrancy",
        "timestamp_dependency":     "TimestampDep",
        "balance_check":            "BalanceCheck",
        "block_number_dependency":  "BlockNumberDep",
        "integer_overflow":         "IntegerOverflow",
        "unchecked_external_call":  "UncheckedCall",
    })

    @property
    def known_class_names(self) -> list:
        """Display names for known classes in label-index order (0..NUM_KNOWN-1)."""
        return [self.CLASS_NAMES[cat] for cat in self.KNOWN_CLASSES]

    @property
    def all_class_names(self) -> list:
        """Display names for all 6 categories in CATEGORIES order."""
        return [self.CLASS_NAMES[cat] for cat in self.CATEGORIES]

    # ==========================================================================
    # Train / Validation / Test Split
    # ==========================================================================
    # Pseudo-unknown: held out entirely from training
    # and used only to evaluate unknown-vulnerability detection performance.
    PSEUDO_UNKNOWN_CLASSES: Tuple[str, ...] = (
        "unchecked_external_call",
    )
    # Known classes used for training the classifier
    KNOWN_CLASSES: Tuple[str, ...] = (
        "reentrancy",
        "timestamp_dependency",
        "integer_overflow",
        "balance_check",
        "block_number_dependency",
    )

    NUM_KNOWN: int = 5  # K in the paper

    TRAIN_RATIO: float = 0.80
    VAL_RATIO: float = 0.10
    TEST_RATIO: float = 0.10
    # Fraction of validation samples to treat as "pseudo-unknown" for threshold calibration
    VAL_PSEUDO_UNKNOWN_FRAC: float = 0.20

    # ==========================================================================
    # Sequence Processing
    # ==========================================================================
    L_MAX: int = 3000       # maximum sequence length (99.2% coverage)
    # For GPU with 16GB VRAM: L_MAX=5000 needs BATCH_SIZE <= 8
    # For GPU with 8GB VRAM: reduce L_MAX to 2000
    VOCAB_MIN_COUNT: int = 1  # include all tokens

    # ==========================================================================
    # Embedding Dimensions
    # ==========================================================================
    # We skip Transformer pre-training (requires 1.2M external traces).
    # Opcode embedding is CBOW-only → 50 dim instead of paper's 114.
    OPCODE_EMB_DIM: int = 50
    PARAM_EMB_DIM: int = 50
    PARAM_MAX_SLOTS: int = 3   # at most 3 atomic types per position (paper: >95% fit)
    NUM_STATS: int = 4         # jump_flag, gas_cost, stack_depth_delta, storage_flag
    USE_PARAM_FEATURES: bool = True   # include parameter-type embeddings in fusion
    USE_STATS_FEATURES: bool = True   # include statistical features in fusion
    USE_GCN_BRANCH: bool = True       # include GCN branch in model

    # Fused feature dimension: 50(opcode) + 50*3(params) + 4(stats) = 204
    @property
    def FUSED_DIM(self) -> int:
        d = self.OPCODE_EMB_DIM
        if self.USE_PARAM_FEATURES:
            d += self.PARAM_EMB_DIM * self.PARAM_MAX_SLOTS
        if self.USE_STATS_FEATURES:
            d += self.NUM_STATS
        return d

    # ==========================================================================
    # Model Architecture
    # ==========================================================================
    CNN_FILTERS: int = 128
    CNN_KERNEL: int = 3
    BILSTM_UNITS: int = 64          # per direction  → output 128

    GCN_HIDDEN: int = 128
    GCN_OUT: int = 64
    GCN_DROPOUT: float = 0.3

    @property
    def BILSTM_OUT(self) -> int:
        return self.BILSTM_UNITS * 2   # 128

    @property
    def CONCAT_DIM(self) -> int:
        d = self.BILSTM_OUT  # 128
        if self.USE_GCN_BRANCH:
            d += self.GCN_OUT  # +64 = 192
        return d

    ATTENTION_HEADS: int = 8
    ATTENTION_DIM: int = 64           # per-head dimension

    FC1_UNITS: int = 64
    FC2_UNITS: int = 32
    DROPOUT1: float = 0.5
    DROPOUT2: float = 0.3

    # ==========================================================================
    # Training (CPU-optimized)
    # ==========================================================================
    BATCH_SIZE: int = 4      # 4 safe for L=5000 on 16GB VRAM (backward pass needs ~2x memory)
    EPOCHS: int = 50
    LEARNING_RATE: float = 1e-3
    LABEL_SMOOTHING: float = 0.1
    PROTOTYPE_LAMBDA: float = 0.01
    WEIGHT_DECAY: float = 1e-5
    EARLY_STOP_PATIENCE: int = 10
    PROTOTYPE_EMA_ALPHA: float = 0.9  # EMA decay for prototype updates

    # ==========================================================================
    # V2 Training Enhancements (contrastive prototypes + outlier exposure)
    # ==========================================================================
    REPULSION_LAMBDA: float = 0.01     # inter-class prototype repulsion weight
    REPULSION_MARGIN: float = 2.0      # minimum distance between prototypes
    OUTLIER_LAMBDA: float = 0.1        # outlier exposure loss weight
    OUTLIER_COUNT: int = 16            # synthetic outliers per batch
    OUTLIER_NOISE: float = 0.15        # Gaussian noise std for outlier generation
    OUTLIER_PROTO_MARGIN: float = 2.0  # min distance from outlier to nearest prototype
    OUTLIER_PROTO_LAMBDA: float = 1.0  # prototype repulsion weight in outlier loss

    # ==========================================================================
    # V5 Training Enhancements (normal traces as negative samples)
    # ==========================================================================
    NORMAL_LAMBDA: float = 0.3         # weight for normal trace loss
    NORMAL_PROTO_MARGIN: float = 2.5   # min distance from normal to nearest prototype
    NORMAL_PROTO_LAMBDA: float = 1.0   # prototype repulsion weight in normal loss

    # ==========================================================================
    # Unknown Detection
    # ==========================================================================
    MC_DROPOUT_T: int = 10           # Monte Carlo forward passes
    TAU_HIGH: float = 0.85           # default, tuned on validation
    TAU_LOW: float = 0.45
    ENTROPY_LOW: float = 0.3
    ENTROPY_HIGH: float = 1.0

    # ==========================================================================
    # Reproducibility
    # ==========================================================================
    SEEDS: Tuple[int, ...] = (42, 43, 44, 45, 46)

    # ==========================================================================
    # Output
    # ==========================================================================
    @property
    def OUTPUT_DIR(self) -> str:
        return os.path.join(self.DATA_ROOT, "models", "output")

    @property
    def MODEL_DIR(self) -> str:
        return os.path.join(self.DATA_ROOT, "models", "checkpoints")

    @property
    def PLOT_DIR(self) -> str:
        return os.path.join(self.OUTPUT_DIR, "plots")

    # ==========================================================================
    # Convenience: category → label index
    # ==========================================================================
    def category_to_label(self, category: str) -> int:
        """Map a category directory name to its numeric label."""
        return self.KNOWN_CLASSES.index(category)

    @property
    def known_class_indices(self) -> Tuple[int, ...]:
        return tuple(range(self.NUM_KNOWN))
