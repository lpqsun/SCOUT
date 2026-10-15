#!/usr/bin/env python3
"""
PyTorch Dataset for smart contract vulnerability detection.
Loads parameter sequences, opcode sequences, dependency graphs,
and precomputed statistical features for training.
"""

import os
import json
import pickle
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class VulnTraceDataset(Dataset):
    """
    Dataset for vulnerability detection traces.

    Each sample includes:
      - param token indices (from param_only.txt)
      - opcode indices (from param_features.json)
      - dependency graph edges (from dep_matrix.json)
      - statistical features (from _stats.npy)
      - label (class index)
      - metadata flags (is_unknown, etc.)
    """

    def __init__(
        self,
        config,
        split: str = "train",
        opc_vocab = None,
        param_vocab = None,
        atom_registry = None,
    ):
        self.config = config
        self.split = split
        self.opc_vocab = opc_vocab
        self.param_vocab = param_vocab
        self.atom_registry = atom_registry
        self.L_max = config.L_MAX

        self.samples: List[dict] = []
        self.class_weights: Optional[torch.Tensor] = None

        self._build_index()

    def _build_index(self):
        """Scan all directories and assign samples to splits."""
        base_dir = self.config.DATA_ROOT
        rng = random.Random(42)  # fixed seed for reproducible splits

        known_samples = defaultdict(list)  # category → list of samples
        unknown_samples = []

        for category in self.config.CATEGORIES:
            param_dir = os.path.join(base_dir, category + self.config.PARAM_DIR_SUFFIX)
            trace_dir = os.path.join(base_dir, category)
            if not os.path.isdir(param_dir):
                continue

            param_files = sorted([
                f for f in os.listdir(param_dir)
                if f.endswith("_param_only.txt")
            ])

            for pf in param_files:
                basename = pf.replace("_param_only.txt", "")
                sample = {
                    "basename": basename,
                    "category": category,
                    "param_dir": param_dir,
                    "trace_dir": trace_dir,
                }

                if category in self.config.PSEUDO_UNKNOWN_CLASSES:
                    unknown_samples.append(sample)
                else:
                    known_samples[category].append(sample)

        # Shuffle known categories
        for cat in known_samples:
            rng.shuffle(known_samples[cat])

        # Split known samples per class
        train_samples, val_samples, test_known = [], [], []
        T = self.config.TRAIN_RATIO
        V = self.config.VAL_RATIO

        for cat, samples in known_samples.items():
            n = len(samples)
            n_train = int(n * T)
            n_val = int(n * V)
            train_samples.extend(samples[:n_train])
            val_samples.extend(samples[n_train:n_train + n_val])
            test_known.extend(samples[n_train + n_val:])

        # Assign based on self.split
        if self.split == "train":
            self.samples = train_samples
        elif self.split == "val":
            self.samples = val_samples
        elif self.split == "test":
            self.samples = test_known + unknown_samples
        else:
            raise ValueError(f"Unknown split: {self.split}")

        # Label each sample
        for s in self.samples:
            if s["category"] in self.config.PSEUDO_UNKNOWN_CLASSES:
                s["label"] = -1
                s["is_unknown"] = True
            else:
                s["label"] = self.config.category_to_label(s["category"])
                s["is_unknown"] = False

        # Shuffle
        rng.shuffle(self.samples)

        # Compute class weights (training set only)
        if self.split == "train":
            label_counts = defaultdict(int)
            for s in self.samples:
                if s["label"] >= 0:
                    label_counts[s["label"]] += 1
            total = sum(label_counts.values())
            K = self.config.NUM_KNOWN
            weights = []
            for k in range(K):
                count = label_counts.get(k, 0)
                if count > 0:
                    w = total / (K * count)
                else:
                    w = 1.0
                weights.append(w)
            self.class_weights = torch.tensor(weights, dtype=torch.float32)

        print(f"  [{self.split}] {len(self.samples)} samples"
              f" (known: {sum(1 for s in self.samples if not s['is_unknown'])}, "
              f"unknown: {sum(1 for s in self.samples if s['is_unknown'])})"
              f"{' class_weights=' + str(self.class_weights.tolist()) if self.class_weights is not None else ''}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_param_seq(self, sample) -> List[str]:
        """Load parameter type token sequence from param_only.txt."""
        path = os.path.join(sample["param_dir"], sample["basename"] + "_param_only.txt")
        tokens = []
        with open(path, 'r') as f:
            for line in f:
                tok = line.strip()
                if tok:
                    tokens.append(tok)
        return tokens

    def _load_opcode_seq(self, sample) -> List[str]:
        """Load opcode sequence from param_features.json."""
        path = os.path.join(sample["param_dir"], sample["basename"] + "_param_features.json")
        with open(path, 'r') as f:
            data = json.load(f)
        return [s["opcode"] for s in data.get("steps", [])]

    def _load_dep_edges(self, sample, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load dependency graph edges from dep_matrix.json.
        Returns (edge_index, edge_attr) truncated to max_len.
        The paper limits each node to k=10 nearest dependencies.
        """
        path = os.path.join(sample["param_dir"], sample["basename"] + "_dep_matrix.json")
        with open(path, 'r') as f:
            dep_dict = json.load(f)

        edges = []
        for src_str, dsts in dep_dict.items():
            src = int(src_str)
            if src >= max_len:
                continue
            # Keep at most 10 dependencies per node (paper constraint)
            for dst in dsts[:10]:
                if dst < max_len:
                    edges.append([src, dst])

        if not edges:
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0,), dtype=torch.float32),
            )

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # (2, E)
        edge_attr = torch.ones(edge_index.shape[1], dtype=torch.float32)     # unweighted
        return edge_index, edge_attr

    def _load_stats(self, sample) -> np.ndarray:
        """Load precomputed statistical features."""
        path = os.path.join(sample["param_dir"], sample["basename"] + "_stats.npy")
        if os.path.exists(path):
            return np.load(path).astype(np.float32)
        # Fallback: zeros
        return np.zeros((1, 4), dtype=np.float32)

    def _pad_or_truncate_1d(self, seq: list, max_len: int, pad_val=0) -> torch.Tensor:
        """Pad or truncate a 1D sequence to max_len."""
        if len(seq) > max_len:
            seq = seq[:max_len]
        else:
            seq = seq + [pad_val] * (max_len - len(seq))
        return torch.tensor(seq, dtype=torch.long)

    def _pad_or_truncate_2d(self, arr: np.ndarray, max_len: int) -> torch.Tensor:
        """Pad or truncate a 2D array (L, D) to (max_len, D)."""
        L, D = arr.shape
        if L > max_len:
            arr = arr[:max_len, :]
        elif L < max_len:
            pad = np.zeros((max_len - L, D), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # Load raw data
        param_tokens = self._load_param_seq(sample)
        opcode_tokens = self._load_opcode_seq(sample)
        stats = self._load_stats(sample)

        orig_len = len(param_tokens)
        L = min(orig_len, self.L_max)

        # Encode parameter tokens to indices
        param_indices = self.param_vocab.encode(param_tokens)
        param_indices = self._pad_or_truncate_1d(param_indices, self.L_max)

        # Encode opcodes to indices
        opc_indices = self.opc_vocab.encode(opcode_tokens)
        opc_indices = self._pad_or_truncate_1d(opc_indices, self.L_max)

        # Stats: (L_max, 4)
        stats_tensor = self._pad_or_truncate_2d(stats, self.L_max)

        # Dependency edges  (truncated to L)
        edge_index, edge_attr = self._load_dep_edges(sample, L)

        # Mask
        mask = torch.zeros(self.L_max, dtype=torch.bool)
        mask[:L] = True

        return {
            "param_indices": param_indices,
            "opc_indices": opc_indices,
            "stats": stats_tensor,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "mask": mask,
            "length": L,
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "is_unknown": sample["is_unknown"],
            "basename": sample["basename"],
            "category": sample["category"],
        }

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        """Collate a list of samples into a batch, padding to max length in batch."""
        batch_max_len = max(item["length"] for item in batch)
        B = len(batch)

        param_indices = torch.stack([item["param_indices"][:batch_max_len] for item in batch])
        opc_indices = torch.stack([item["opc_indices"][:batch_max_len] for item in batch])
        stats = torch.stack([item["stats"][:batch_max_len, :] for item in batch])
        mask = torch.stack([item["mask"][:batch_max_len] for item in batch])
        labels = torch.stack([item["label"] for item in batch])
        is_unknown = torch.tensor([item["is_unknown"] for item in batch], dtype=torch.bool)

        # Batch graphs: offset node indices by sample position
        edge_indices_list = []
        edge_attrs_list = []
        for i, item in enumerate(batch):
            ei = item["edge_index"].clone()
            if ei.numel() > 0:
                ei[0, :] += i * batch_max_len  # source offset
                ei[1, :] += i * batch_max_len  # target offset
            edge_indices_list.append(ei)
            edge_attrs_list.append(item["edge_attr"])

        edge_index = torch.cat(edge_indices_list, dim=1) if edge_indices_list else torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.cat(edge_attrs_list, dim=0) if edge_attrs_list else torch.zeros((0,), dtype=torch.float32)

        basenames = [item["basename"] for item in batch]
        categories = [item["category"] for item in batch]
        lengths = [item["length"] for item in batch]

        return {
            "param_indices": param_indices,
            "opc_indices": opc_indices,
            "stats": stats,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "mask": mask,
            "labels": labels,
            "is_unknown": is_unknown,
            "basename": basenames,
            "category": categories,
            "length": lengths,
            "batch_size": B,
            "batch_max_len": batch_max_len,
        }


def create_dataloaders(config, opc_vocab, param_vocab, atom_registry):
    """Create train, validation, and test dataloaders."""
    train_ds = VulnTraceDataset(config, split="train",
                                opc_vocab=opc_vocab, param_vocab=param_vocab,
                                atom_registry=atom_registry)
    val_ds = VulnTraceDataset(config, split="val",
                              opc_vocab=opc_vocab, param_vocab=param_vocab,
                              atom_registry=atom_registry)
    test_ds = VulnTraceDataset(config, split="test",
                               opc_vocab=opc_vocab, param_vocab=param_vocab,
                               atom_registry=atom_registry)

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        collate_fn=VulnTraceDataset.collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=VulnTraceDataset.collate_fn, num_workers=0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=VulnTraceDataset.collate_fn, num_workers=0,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_dataset": train_ds,
        "val_dataset": val_ds,
        "test_dataset": test_ds,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import Config
    from vocabulary import build_all_vocabs

    cfg = Config()
    opc_vocab, param_vocab, atom_registry = build_all_vocabs(cfg)
    loaders = create_dataloaders(cfg, opc_vocab, param_vocab, atom_registry)

    # Verify
    for split_name in ["train", "val", "test"]:
        loader = loaders[split_name]
        ds = loaders[f"{split_name}_dataset"]
        batch = next(iter(loader))
        print(f"\n  {split_name}: {len(ds)} samples, batch shapes:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"    {k}: {list(v.shape)}")
        if ds.class_weights is not None:
            print(f"    class_weights: {ds.class_weights.tolist()}")
