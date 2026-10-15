#!/usr/bin/env python3
"""
CBOW embedding training for opcodes and parameter types.
Manual PyTorch implementation (avoids gensim dependency).

Section 4.4.1 of the paper: CBOW with d=50, window=5, negative sampling.
"""

import os
import random
from collections import Counter
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset


# ==============================================================================
# CBOW Training
# ==============================================================================

class CBOWDataset(IterableDataset):
    """Streaming dataset that yields (context, target) pairs for CBOW training."""

    def __init__(self, sequences: List[List[int]], window: int = 5,
                 neg_samples: int = 5, vocab_size: int = 100):
        self.sequences = sequences
        self.window = window
        self.neg_samples = neg_samples
        self.vocab_size = vocab_size
        # Build noise distribution for negative sampling (unigram ^ 0.75)
        # Exclude PAD(0) and UNK(1) from negative samples
        counter = Counter()
        for seq in sequences:
            counter.update(seq)
        freq = np.zeros(vocab_size, dtype=np.float64)
        for token, count in counter.items():
            if token >= 2 and token < vocab_size:
                freq[token] = count
        freq = freq ** 0.75
        total = freq.sum()
        if total > 0:
            self.noise_dist = freq / total
        else:
            self.noise_dist = np.ones(vocab_size, dtype=np.float64) / vocab_size

    def __iter__(self):
        for seq in self.sequences:
            if len(seq) < 2:
                continue
            for i, target in enumerate(seq):
                start = max(0, i - self.window)
                end = min(len(seq), i + self.window + 1)
                context = [seq[j] for j in range(start, end) if j != i and seq[j] != 0]
                if not context:
                    continue
                # Sample a fixed number of context words
                if len(context) > 4:
                    context = random.sample(context, 4)
                # Pad to 4
                while len(context) < 4:
                    context.append(0)
                # Negative samples
                negs = list(np.random.choice(
                    self.vocab_size, size=self.neg_samples,
                    p=self.noise_dist, replace=False,
                ))
                yield {
                    "context": torch.tensor(context, dtype=torch.long),
                    "target": torch.tensor(target, dtype=torch.long),
                    "negatives": torch.tensor(negs, dtype=torch.long),
                }


class CBOWModel(nn.Module):
    """Manual CBOW: embedding lookup → mean pooling → linear → softmax."""

    def __init__(self, vocab_size: int, embed_dim: int = 50):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.output = nn.Linear(embed_dim, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embeddings.weight)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        context: (B, ctx_size)  — context word indices
        Returns: (B, vocab_size) — logits for target word
        """
        emb = self.embeddings(context)         # (B, ctx_size, dim)
        pooled = emb.mean(dim=1)               # (B, dim)
        return self.output(pooled)             # (B, vocab_size)


def _precompute_cbow_pairs(sequences, window, neg_samples, vocab_size, noise_dist):
    """Pre-compute all CBOW training pairs ONCE as tensors (GPU-friendly)."""
    all_context = []
    all_target = []
    all_negatives = []

    for seq in sequences:
        L = len(seq)
        if L < 2:
            continue
        # Build a fast lookup: for each position, find context words
        for i, target in enumerate(seq):
            if target == 0:  # skip PAD
                continue
            start = max(0, i - window)
            end = min(L, i + window + 1)
            context = [seq[j] for j in range(start, end) if j != i and seq[j] != 0]
            if not context:
                continue
            # Sample up to 4 context words
            if len(context) > 4:
                context = random.sample(context, 4)
            while len(context) < 4:
                context.append(0)  # PAD

            negs = list(np.random.choice(vocab_size, size=neg_samples,
                                         p=noise_dist, replace=False))

            all_context.append(context)
            all_target.append(target)
            all_negatives.append(negs)

    context_t = torch.tensor(all_context, dtype=torch.long)
    target_t = torch.tensor(all_target, dtype=torch.long)
    negatives_t = torch.tensor(all_negatives, dtype=torch.long)
    return context_t, target_t, negatives_t


def train_cbow(
    sequences: List[List[int]],
    vocab_size: int,
    embed_dim: int = 50,
    window: int = 5,
    neg_samples: int = 5,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 4096,
    device: str = "cpu",
) -> np.ndarray:
    """
    Train a CBOW model and return the embedding matrix.
    Pre-computes all training pairs once for fast GPU training.
    """
    print(f"  Training CBOW: vocab={vocab_size}, dim={embed_dim}, "
          f"epochs={epochs}, seqs={len(sequences)}")

    # Build noise distribution
    counter = Counter()
    for seq in sequences:
        counter.update(seq)
    freq = np.zeros(vocab_size, dtype=np.float64)
    for token, count in counter.items():
        if token >= 2 and token < vocab_size:
            freq[token] = count
    freq = freq ** 0.75
    total = freq.sum()
    noise_dist = freq / total if total > 0 else np.ones(vocab_size) / vocab_size

    # Pre-compute all pairs ONCE (CPU, but only once)
    print(f"    Pre-computing training pairs...")
    context, target, negatives = _precompute_cbow_pairs(
        sequences, window, neg_samples, vocab_size, noise_dist)
    n_pairs = len(target)
    print(f"    {n_pairs:,} pairs generated, moving to {device}...")

    # Move to GPU
    context = context.to(device)
    target = target.to(device)
    negatives = negatives.to(device)

    # TensorDataset + DataLoader (fast GPU feeding)
    dataset = torch.utils.data.TensorDataset(context, target, negatives)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CBOWModel(vocab_size, embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for ctx, tgt, neg in loader:
            logits = model(ctx)
            pos_scores = -F.logsigmoid(logits.gather(1, tgt.unsqueeze(1)))
            neg_scores = -F.logsigmoid(-logits.gather(1, neg)).sum(dim=1)
            loss = pos_scores.mean() + neg_scores.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}, loss={total_loss/n_batches:.4f}")

    weights = model.embeddings.weight.detach().cpu().numpy()
    return weights.astype(np.float32)


# ==============================================================================
# Hybrid Embedding Layer
# ==============================================================================

class HybridEmbedding(nn.Module):
    """
    Frozen embedding layer that combines:
      - Opcode embedding (pretrained CBOW)
      - Parameter type embedding (pretrained CBOW, up to 3 slots)
      - Statistical features (4 scalar, passed through directly)

    Output dimension = opc_dim + param_dim * 3 + 4  (default 204)
    """

    def __init__(self, config, opc_emb_weights: np.ndarray,
                 param_emb_weights: np.ndarray):
        super().__init__()
        self.config = config
        self.opc_dim = config.OPCODE_EMB_DIM
        self.param_dim = config.PARAM_EMB_DIM
        self.max_slots = config.PARAM_MAX_SLOTS
        self.fused_dim = config.FUSED_DIM

        # Frozen embedding layers
        opc_vocab_size, edim = opc_emb_weights.shape
        self.opc_embed = nn.Embedding(opc_vocab_size, edim, padding_idx=0)
        self.opc_embed.weight.data.copy_(torch.from_numpy(opc_emb_weights))
        self.opc_embed.weight.requires_grad = False

        atom_vocab_size, pdim = param_emb_weights.shape
        self.param_embed = nn.Embedding(atom_vocab_size, pdim, padding_idx=0)
        self.param_embed.weight.data.copy_(torch.from_numpy(param_emb_weights))
        self.param_embed.weight.requires_grad = False

        # Dummy embedding for unknown atoms (maps to zero vector)
        self.zero_pad = nn.Parameter(torch.zeros(1, self.param_dim), requires_grad=False)

    def forward(self, opc_indices: torch.Tensor,
                param_indices: torch.Tensor,
                stats: torch.Tensor,
                atom_registry) -> torch.Tensor:
        """
        Args:
            opc_indices:   (B, L) — opcode token indices
            param_indices: (B, L) — param token indices (compound tokens)
            stats:         (B, L, 4) — precomputed statistical features
            atom_registry: AtomicRegistry for parsing compound tokens

        Returns:
            fused: (B, L, fused_dim) mixed feature vectors
        """
        B, L = opc_indices.shape
        parts = []

        # 1. Opcode embedding: (B, L, opc_dim) — always included
        parts.append(self.opc_embed(opc_indices))

        # 2. Parameter embedding (conditional)
        if self.config.USE_PARAM_FEATURES:
            if self._atom_lookup is None:
                raise RuntimeError("Atom lookup table not built. Call build_atom_lookup() first.")
            lut = self._atom_lookup.to(opc_indices.device)
            safe_idx = param_indices.clamp(0, lut.size(0) - 1)
            atom_indices = lut[safe_idx]  # (B, L, max_slots)
            # PAD(0) and UNK(1) compound tokens map to zero entries in the lookup;
            # zero out their atom embeddings to avoid leaking VALUE semantics.
            invalid_mask = (param_indices <= 1).unsqueeze(-1).unsqueeze(-1).float()  # (B, L, 1, 1)
            atom_emb = self.param_embed(atom_indices)  # (B, L, max_slots, param_dim)
            atom_emb = atom_emb * (1.0 - invalid_mask)  # zero out PAD / UNK positions
            atom_emb = atom_emb.reshape(B, L, self.max_slots * self.param_dim)
            parts.append(atom_emb)

        # 3. Statistical features (conditional)
        if self.config.USE_STATS_FEATURES:
            parts.append(stats)

        fused = torch.cat(parts, dim=-1)  # (B, L, fused_dim)
        return fused

    _atom_lookup = None  # Will be set by build_atom_lookup()

    def build_atom_lookup(self, param_vocab, atom_registry):
        """Build (param_vocab_size, max_slots) lookup: compound token → atomic indices."""
        import torch
        V = len(param_vocab)  # includes PAD(0) and UNK(1)
        lut = torch.zeros(V, self.max_slots, dtype=torch.long)
        for token_str, token_idx in param_vocab.token_to_idx.items():
            atoms = atom_registry.parse_token(token_str)
            for j, a in enumerate(atoms[:self.max_slots]):
                if a < self.param_embed.num_embeddings:
                    lut[token_idx, j] = a
        self._atom_lookup = lut
        return lut


def train_all_embeddings(config, opc_vocab, param_vocab, atom_registry,
                         train_loader) -> HybridEmbedding:
    """Train CBOW embeddings and create the HybridEmbedding layer."""
    device = config.DEVICE

    # Collect sequences from training set
    print("\n  Collecting training sequences for CBOW...")
    opc_seqs = []
    param_seqs = []
    for batch in train_loader:
        for b in range(batch["batch_size"]):
            L_val = batch["length"][b]
            L = L_val.item() if hasattr(L_val, 'item') else int(L_val)
            if L > 0:
                opc_seqs.append(batch["opc_indices"][b, :L].tolist())
                param_seqs.append(batch["param_indices"][b, :L].tolist())

    print(f"  Collected {len(opc_seqs)} sequences")

    # Train CBOW for opcodes
    print("\n  Training opcode CBOW...")
    opc_weights = train_cbow(
        opc_seqs, vocab_size=len(opc_vocab),
        embed_dim=config.OPCODE_EMB_DIM, epochs=10, device=device,
    )

    # Train CBOW for parameter types (on ATOMIC level)
    print("\n  Building atomic-level param sequences...")
    # We need atom-level sequences for CBOW training
    atom_seqs = []
    for seq_idx in param_seqs:
        atom_seq = []
        for tok_idx in seq_idx:
            token_str = param_vocab.idx_to_token.get(tok_idx, "NULL")
            atoms = atom_registry.parse_token(token_str)
            atom_seq.extend(atoms)
        if atom_seq:
            atom_seqs.append(atom_seq)

    print(f"  Collected {len(atom_seqs)} atomic sequences")

    print("\n  Training parameter CBOW...")
    param_weights = train_cbow(
        atom_seqs, vocab_size=len(atom_registry),
        embed_dim=config.PARAM_EMB_DIM, epochs=10, device=device,
    )

    hybrid = HybridEmbedding(config, opc_weights, param_weights)
    hybrid.build_atom_lookup(param_vocab, atom_registry)
    return hybrid
