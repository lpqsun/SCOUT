#!/usr/bin/env python3
"""
Stage 1: One-Class Deep Anomaly Detection (Deep SVDD).
Section 4.7 of the paper.

Filters normal (non-vulnerable) contracts from anomalous (potentially vulnerable)
contracts BEFORE the Stage 2 known/unknown classifier.

Architecture:
  - Shares feature extraction backbone with Stage 2 (frozen HybridEmbedding)
  - Lightweight encoder: 1D CNN → GlobalAvgPool → FC(128→64→32)
  - Learns a compact hypersphere: center c, radius R
  - Anomaly score: s(x) = ||φ(x) - c||²
  - Threshold δ = 95th percentile of normal training distances

Usage:
  When NORMAL contract traces are available, train with:
    model = DeepSVDD(config)
    model.fit(normal_train_loader, hybrid_emb, atom_registry)
    predictions = model.predict(test_loader)
"""

import os
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm


# ==============================================================================
# Deep SVDD Encoder
# ==============================================================================

class SVDDEncoder(nn.Module):
    """
    Lightweight encoder for one-class anomaly detection.
    Architecture: 1D CNN → GlobalAvgPool → FC(128→64→32)
    Output: 32-dim feature vector for hypersphere projection.
    """

    def __init__(self, input_dim: int, hidden_dims=(128, 64), output_dim=32,
                 dropout=0.3):
        super().__init__()
        self.input_dim = input_dim

        # 1D CNN for local pattern extraction
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=3, padding='same')
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding='same')
        self.conv_drop = nn.Dropout(dropout)

        # FC layers
        self.fc1 = nn.Linear(64, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], output_dim)
        self.fc_drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, L, input_dim) fused feature tensor (from HybridEmbedding)
        Returns:
            z: (B, output_dim) latent representation
        """
        # CNN expects (B, C, L)
        x = x.transpose(1, 2)  # (B, input_dim, L)
        x = F.relu(self.conv1(x))
        x = self.conv_drop(x)
        x = F.relu(self.conv2(x))
        x = self.conv_drop(x)

        # Global average pooling over sequence dimension
        x = x.mean(dim=2)  # (B, 64)

        # FC
        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        z = self.fc2(x)  # (B, output_dim) — NO activation on output

        return z


# ==============================================================================
# Deep SVDD Model
# ==============================================================================

class DeepSVDD:
    """
    One-Class Deep SVDD anomaly detector.

    Training:
      1. Forward all normal samples through frozen HybridEmbedding + SVDDEncoder
      2. Compute center c = mean of all latent vectors
      3. Optimize encoder to minimize ||φ(x) - c||² + weight decay

    Inference:
      s(x) = ||φ(x) - c||²
      anomaly if s(x) > δ
    """

    def __init__(self, config):
        self.config = config
        self.input_dim = config.FUSED_DIM  # from HybridEmbedding
        self.output_dim = config.FC2_UNITS  # 32-dim latent
        self.device = config.DEVICE

        self.encoder = SVDDEncoder(
            input_dim=self.input_dim,
            hidden_dims=(config.FC1_UNITS, config.FC2_UNITS),
            output_dim=self.output_dim,
            dropout=config.DROPOUT2,
        ).to(self.device)

        self.center = None       # torch.Tensor (output_dim,) — fixed after init
        self.threshold = None    # float — 95th percentile of normal distances
        self._fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, normal_loader, hybrid_emb, atom_registry,
            epochs=30, lr=1e-3, weight_decay=1e-5, nu=0.1, verbose=True):
        """
        Train the SVDD encoder on NORMAL (non-vulnerable) samples only.

        Args:
            normal_loader: DataLoader yielding only normal contract batches
            hybrid_emb: frozen HybridEmbedding from embeddings.py
            atom_registry: AtomicRegistry
            epochs: training epochs
            lr: learning rate
            weight_decay: L2 regularization (λ in the paper)
            nu: center update EMA factor (paper uses fixed center; nu=0 freezes it)

        Returns:
            self
        """
        # Step 1: Initialize center c
        if verbose:
            print("\n  [SVDD] Initializing center c...")
        self._init_center(normal_loader, hybrid_emb, atom_registry)

        # Step 2: Optimize encoder
        if verbose:
            print(f"  [SVDD] Training encoder ({epochs} epochs, lr={lr})...")
        optimizer = AdamW(self.encoder.parameters(), lr=lr,
                         weight_decay=weight_decay)

        self.encoder.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in normal_loader:
                fused = self._build_fused(batch, hybrid_emb, atom_registry)
                z = self.encoder(fused)  # (B, output_dim)

                # SVDD loss: mean squared distance to center
                # L = 1/N Σ ||φ(x_i) - c||²
                dist_sq = ((z - self.center.unsqueeze(0)) ** 2).sum(dim=1)
                loss = dist_sq.mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                print(f"    Epoch {epoch+1:3d}/{epochs}, loss={avg_loss:.6f}")

        # Step 3: Compute threshold δ
        if verbose:
            print("  [SVDD] Computing anomaly threshold δ...")
        self._compute_threshold(normal_loader, hybrid_emb, atom_registry,
                               percentile=95)

        self._fitted = True
        if verbose:
            print(f"  [SVDD] Training complete. Center norm={self.center.norm():.4f}, "
                  f"δ={self.threshold:.4f}")

        return self

    def _init_center(self, loader, hybrid_emb, atom_registry):
        """Initialize center c as mean of latent vectors from normal samples."""
        self.encoder.eval()
        all_z = []
        with torch.no_grad():
            for batch in loader:
                fused = self._build_fused(batch, hybrid_emb, atom_registry)
                z = self.encoder(fused)
                all_z.append(z.cpu())

        all_z = torch.cat(all_z, dim=0)  # (N, output_dim)
        self.center = all_z.mean(dim=0).to(self.device)

        # Small perturbation to avoid trivial solution (c ≠ 0)
        # Standard SVDD practice: add epsilon noise
        if self.center.norm() < 1e-6:
            self.center += 0.01 * torch.randn_like(self.center)

    def _compute_threshold(self, loader, hybrid_emb, atom_registry,
                          percentile=95):
        """Set δ as the given percentile of distances on normal samples."""
        self.encoder.eval()
        all_dists = []
        with torch.no_grad():
            for batch in loader:
                fused = self._build_fused(batch, hybrid_emb, atom_registry)
                z = self.encoder(fused)
                dist_sq = ((z - self.center.unsqueeze(0)) ** 2).sum(dim=1)
                all_dists.append(dist_sq.cpu())

        all_dists = torch.cat(all_dists, dim=0).numpy()
        self.threshold = float(np.percentile(all_dists, percentile))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, loader, hybrid_emb, atom_registry):
        """
        Predict anomaly labels for a data loader.

        Returns:
            list of dicts: [{anomaly_score, is_anomaly, ...}, ...]
        """
        if not self._fitted:
            raise RuntimeError("DeepSVDD not fitted. Call .fit() first.")

        self.encoder.eval()
        results = []

        for batch in tqdm(loader, desc="  SVDD predict"):
            fused = self._build_fused(batch, hybrid_emb, atom_registry)
            z = self.encoder(fused)
            dist_sq = ((z - self.center.unsqueeze(0)) ** 2).sum(dim=1)

            B = dist_sq.size(0)
            for i in range(B):
                score = dist_sq[i].item()
                is_anomaly = score > self.threshold
                results.append({
                    "svdd_score": score,
                    "svdd_is_anomaly": bool(is_anomaly),
                    "svdd_threshold": self.threshold,
                    "basename": batch.get("basename", [""])[i] if "basename" in batch else "",
                    "category": batch.get("category", [""])[i] if "category" in batch else "",
                    "is_unknown": batch.get("is_unknown", [False])[i].item()
                        if "is_unknown" in batch else False,
                })

        return results

    def get_latent_vectors(self, loader, hybrid_emb, atom_registry):
        """Extract latent vectors for all samples (for t-SNE visualization)."""
        self.encoder.eval()
        all_z = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                fused = self._build_fused(batch, hybrid_emb, atom_registry)
                z = self.encoder(fused)
                all_z.append(z.cpu().numpy())
                all_labels.append(batch["labels"].cpu().numpy())

        return np.concatenate(all_z, axis=0), np.concatenate(all_labels, axis=0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_fused(self, batch, hybrid_emb, atom_registry):
        """Build fused features from batch using HybridEmbedding."""
        opc = batch["opc_indices"].to(self.device)
        param = batch["param_indices"].to(self.device)
        stats = batch["stats"].to(self.device)
        return hybrid_emb(opc, param, stats, atom_registry)

    # ------------------------------------------------------------------
    # Batch prediction (used by TwoStageDetector)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_batch(self, batch, hybrid_emb, atom_registry):
        """
        Predict anomaly scores for a single batch.
        Used by TwoStageDetector for integrated two-stage inference.

        Args:
            batch: collated batch dict from VulnTraceDataset
            hybrid_emb: HybridEmbedding layer
            atom_registry: AtomicRegistry

        Returns:
            list of dicts: [{svdd_score, svdd_is_anomaly}, ...]
        """
        if not self._fitted:
            # Not fitted — pass everything through as anomalous
            B = batch["labels"].size(0)
            return [{"svdd_score": 0.0, "svdd_is_anomaly": True}
                    for _ in range(B)]

        self.encoder.eval()
        fused = self._build_fused(batch, hybrid_emb, atom_registry)
        z = self.encoder(fused)
        dist_sq = ((z - self.center.unsqueeze(0)) ** 2).sum(dim=1)

        B = dist_sq.size(0)
        return [
            {
                "svdd_score": dist_sq[i].item(),
                "svdd_is_anomaly": bool(dist_sq[i].item() > self.threshold),
            }
            for i in range(B)
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save SVDD model state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "encoder_state": self.encoder.state_dict(),
            "center": self.center.cpu() if self.center is not None else None,
            "threshold": self.threshold,
            "config": self.config,
        }
        torch.save(state, path)

    def load(self, path: str, map_location="cpu"):
        """Load SVDD model state."""
        state = torch.load(path, map_location=map_location)
        self.encoder.load_state_dict(state["encoder_state"])
        self.center = state["center"].to(self.device) if state["center"] is not None else None
        self.threshold = state["threshold"]
        self._fitted = True
        return self


# ==============================================================================
# Two-Stage Pipeline Wrapper
# ==============================================================================

class TwoStageDetector:
    """
    Complete two-stage vulnerability detection pipeline.

    Stage 1: DeepSVDD filters normal contracts
    Stage 2: VulnDetectionModel classifies anomaly as known/unknown vulnerability

    Section 4.7 + 4.8 of the paper.
    """

    def __init__(self, config, stage2_model, prototypes,
                 hybrid_emb, atom_registry,
                 svdd_model=None):
        self.config = config
        self.stage2_model = stage2_model
        self.prototypes = prototypes
        self.hybrid_emb = hybrid_emb
        self.atom_registry = atom_registry
        self.svdd = svdd_model  # None if Stage 1 is skipped
        self.device = config.DEVICE

    @torch.no_grad()
    def predict(self, loader, tau_high=None, tau_low=None,
                H_low=None, H_high=None, delta_max=None,
                skip_stage1=False):
        """
        Run two-stage detection.

        Returns:
            list of dicts with keys:
              stage1_anomaly, stage2_decision, stage2_max_prob,
              stage2_entropy, stage2_min_dist, final_decision
        """
        from detect import mc_dropout_forward, compute_prototype_distances

        if tau_high is None:
            tau_high = self.config.TAU_HIGH
        if tau_low is None:
            tau_low = self.config.TAU_LOW
        if H_low is None:
            H_low = self.config.ENTROPY_LOW
        if H_high is None:
            H_high = self.config.ENTROPY_HIGH

        all_results = []

        for batch in tqdm(loader, desc="  Two-Stage"):
            B = batch["labels"].size(0)

            # Stage 1: SVDD anomaly detection
            stage1_scores = None
            stage1_anomalies = None
            if self.svdd is not None and not skip_stage1:
                svdd_results = self.svdd.predict_batch(
                    batch, self.hybrid_emb, self.atom_registry)
                stage1_scores = [s["svdd_score"] for s in svdd_results]
                stage1_anomalies = [s["svdd_is_anomaly"] for s in svdd_results]

            # Stage 2: Known/Unknown classification (on all samples)
            mc = mc_dropout_forward(self.stage2_model, batch, self.device,
                                   T=self.config.MC_DROPOUT_T)
            min_dist, nearest_class = compute_prototype_distances(
                mc["penult_features"], self.prototypes)

            for i in range(B):
                mp = mc["max_prob"][i].item()
                ent = mc["entropy"][i].item()
                md = min_dist[i].item()
                pc = mc["pred_class"][i].item()

                # Stage 2 decision
                known_cond = (mp >= tau_high and ent <= H_low)
                if delta_max is not None:
                    known_cond = known_cond and (md <= delta_max)

                unknown_cond = (mp < tau_low or ent >= H_high)
                if delta_max is not None:
                    unknown_cond = unknown_cond or (md > delta_max)

                if known_cond:
                    s2_decision = "known"
                elif unknown_cond:
                    s2_decision = "unknown"
                else:
                    s2_decision = "ambiguous"

                # Final decision combining both stages
                is_stage1_anomaly = stage1_anomalies[i] if stage1_anomalies else True
                if not is_stage1_anomaly and self.svdd is not None:
                    final_decision = "normal"
                else:
                    final_decision = s2_decision

                all_results.append({
                    "true_label": batch["labels"][i].item(),
                    "is_unknown": batch["is_unknown"][i].item(),
                    "category": batch.get("category", [""])[i] if "category" in batch else "",
                    "basename": batch["basename"][i] if "basename" in batch else "",
                    "stage1_score": stage1_scores[i] if stage1_scores else None,
                    "stage1_anomaly": is_stage1_anomaly,
                    "stage2_max_prob": mp,
                    "stage2_entropy": ent,
                    "stage2_min_dist": md,
                    "stage2_pred_class": pc,
                    "stage2_decision": s2_decision,
                    "final_decision": final_decision,
                })

        return all_results


# ==============================================================================
# Evaluation helpers
# ==============================================================================

def evaluate_stage1(svdd_results, ground_truth_is_vulnerable):
    """
    Evaluate Stage 1 (anomaly detection) performance.

    Args:
        svdd_results: list from DeepSVDD.predict()
        ground_truth_is_vulnerable: list of bool (True=vulnerable, False=normal)

    Returns:
        dict with precision, recall, f1, accuracy
    """
    y_true = np.array(ground_truth_is_vulnerable, dtype=bool)
    y_pred = np.array([r["svdd_is_anomaly"] for r in svdd_results], dtype=bool)

    tp = (y_true & y_pred).sum()
    fp = ((~y_true) & y_pred).sum()
    fn = (y_true & (~y_pred)).sum()
    tn = ((~y_true) & (~y_pred)).sum()

    prec = tp / (tp + fp + 1e-10)
    rec = tp / (tp + fn + 1e-10)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    acc = (tp + tn) / len(y_true)

    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "accuracy": float(acc),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import Config
    from vocabulary import build_all_vocabs
    from dataset import create_dataloaders

    cfg = Config()
    print(f"Device: {cfg.DEVICE_NAME}")
    print("Stage 1 SVDD module ready.")
    print(f"  SVDDEncoder input_dim={cfg.FUSED_DIM}, output_dim={cfg.FC2_UNITS}")
    print(f"  NOTE: Requires NORMAL contract traces for training.")
    print(f"  Currently only vulnerability traces are available.")
