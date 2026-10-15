#!/usr/bin/env python3
"""
Vocabulary building for opcodes and parameter type tokens.
Produces mappings between strings and integer indices for the model.
"""

import os
import json
import pickle
from collections import Counter
from typing import List, Dict, Tuple, Optional


# ==============================================================================
# Atomic Parameter Types  (corresponding to PARAM_TYPE_MAP in param_feature_simulator.py)
# ==============================================================================
ATOMIC_PARAM_TYPES = [
    "VALUE",
    "ACCOUNTADDRESS",
    "ORIGINADDRESS",
    "CALLERADDRESS",
    "COINBASEADDRESS",
    "ETHVALUE",
    "INPUTDATASIZE",
    "OUTPUTDATASIZE",
    "BLOCKTIMESTAMP",
    "BLOCKNUMBER",
    "BLOCKRANDAO",
    "BLOCKDIFFICULTY",
    "ETHCHAINID",
    "COUNTER",
    "RUNCODESIZE",
    "MEMORYSIZE",
    "BLOCKGASLIMIT",
    "GASCOST",
    "GASPRICE",
    "ACCOUNTBALANCE",
    "BASECOST",
    "NULL",
]

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class AtomicRegistry:
    """Registry mapping atomic parameter type strings to indices."""

    def __init__(self):
        self.atom_to_idx: Dict[str, int] = {}
        self.idx_to_atom: Dict[int, str] = {}
        for i, atom in enumerate(ATOMIC_PARAM_TYPES):
            self.atom_to_idx[atom] = i
            self.idx_to_atom[i] = atom

    def __len__(self) -> int:
        return len(self.atom_to_idx)

    def encode(self, atom: str) -> int:
        """Map an atomic type string to its index."""
        return self.atom_to_idx.get(atom, self.atom_to_idx["NULL"])

    def decode(self, idx: int) -> str:
        return self.idx_to_atom.get(idx, "NULL")

    def parse_token(self, token: str) -> List[int]:
        """
        Parse a parameter token string into a list of atomic type indices.

        Compound token "(ACCOUNTADDRESS VALUE)" → [idx(ACCOUNTADDRESS), idx(VALUE)]
        Single token "VALUE" → [idx(VALUE)]
        "NULL" → []
        """
        token = token.strip()
        if token == "NULL" or token == "":
            return []
        if token.startswith("(") and token.endswith(")"):
            inner = token[1:-1]
            atoms = inner.split()
            return [self.encode(a) for a in atoms]
        return [self.encode(token)]


class OpcVocab:
    """Vocabulary for opcode mnemonics."""

    def __init__(self):
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}
        self._frozen = False

    def build(self, data_dirs: List[str], min_count: int = 1):
        """Scan all param_features.json files to collect opcode tokens."""
        counter = Counter()
        for data_dir in data_dirs:
            if not os.path.isdir(data_dir):
                continue
            for fname in os.listdir(data_dir):
                if fname.endswith("_param_features.json"):
                    fpath = os.path.join(data_dir, fname)
                    try:
                        with open(fpath, 'r') as f:
                            data = json.load(f)
                        for step in data.get("steps", []):
                            counter[step.get("opcode", "UNK")] += 1
                    except (json.JSONDecodeError, KeyError, FileNotFoundError):
                        continue

        # Reserve 0 for PAD, 1 for UNK
        self.token_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx_to_token = {0: PAD_TOKEN, 1: UNK_TOKEN}
        idx = 2
        for token, count in sorted(counter.items()):
            if count >= min_count:
                self.token_to_idx[token] = idx
                self.idx_to_token[idx] = token
                idx += 1
        self._frozen = True

    def __len__(self) -> int:
        return len(self.token_to_idx)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_idx.get(t, 1) for t in tokens]  # 1=UNK

    def decode(self, indices: List[int]) -> List[str]:
        return [self.idx_to_token.get(i, UNK_TOKEN) for i in indices]

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({"token_to_idx": self.token_to_idx,
                         "idx_to_token": self.idx_to_token}, f)

    @classmethod
    def load(cls, path: str) -> "OpcVocab":
        with open(path, 'rb') as f:
            data = pickle.load(f)
        vocab = cls()
        vocab.token_to_idx = data["token_to_idx"]
        vocab.idx_to_token = data["idx_to_token"]
        vocab._frozen = True
        return vocab


class ParamVocab:
    """Vocabulary for parameter type tokens (both atomic and compound)."""

    def __init__(self):
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}
        self._frozen = False

    def build(self, data_dirs: List[str], min_count: int = 1):
        """Scan all param_only.txt files to collect unique parameter type tokens."""
        counter = Counter()
        for data_dir in data_dirs:
            if not os.path.isdir(data_dir):
                continue
            for fname in os.listdir(data_dir):
                if fname.endswith("_param_only.txt"):
                    fpath = os.path.join(data_dir, fname)
                    try:
                        with open(fpath, 'r') as f:
                            for line in f:
                                token = line.strip()
                                if token:
                                    counter[token] += 1
                    except FileNotFoundError:
                        continue

        self.token_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx_to_token = {0: PAD_TOKEN, 1: UNK_TOKEN}
        idx = 2
        for token, count in sorted(counter.items()):
            if count >= min_count:
                self.token_to_idx[token] = idx
                self.idx_to_token[idx] = token
                idx += 1
        self._frozen = True

    def __len__(self) -> int:
        return len(self.token_to_idx)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_idx.get(t, 1) for t in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        return [self.idx_to_token.get(i, UNK_TOKEN) for i in indices]

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({"token_to_idx": self.token_to_idx,
                         "idx_to_token": self.idx_to_token}, f)

    @classmethod
    def load(cls, path: str) -> "ParamVocab":
        with open(path, 'rb') as f:
            data = pickle.load(f)
        vocab = cls()
        vocab.token_to_idx = data["token_to_idx"]
        vocab.idx_to_token = data["idx_to_token"]
        vocab._frozen = True
        return vocab


def build_all_vocabs(config) -> Tuple[OpcVocab, ParamVocab, AtomicRegistry]:
    """
    Build opcode vocabulary, parameter vocabulary, and atomic registry
    from all training data directories.
    """
    from config import Config
    cfg = config

    # Gather all param feature directories
    data_dirs = [
        os.path.join(cfg.DATA_ROOT, cat + cfg.PARAM_DIR_SUFFIX)
        for cat in cfg.CATEGORIES
    ]

    print(f"Building vocabularies from {len(data_dirs)} directories...")

    print("  Building opcode vocabulary...")
    opc_vocab = OpcVocab()
    opc_vocab.build(data_dirs, min_count=cfg.VOCAB_MIN_COUNT)
    print(f"    OpcVocab size: {len(opc_vocab)}")

    print("  Building parameter type vocabulary...")
    param_vocab = ParamVocab()
    param_vocab.build(data_dirs, min_count=cfg.VOCAB_MIN_COUNT)
    print(f"    ParamVocab size: {len(param_vocab)}")

    print("  Building atomic type registry...")
    atom_registry = AtomicRegistry()
    print(f"    Atomic types: {len(atom_registry)}")

    # Save
    save_dir = os.path.join(cfg.DATA_ROOT, "models")
    os.makedirs(save_dir, exist_ok=True)
    opc_vocab.save(os.path.join(save_dir, "opc_vocab.pkl"))
    param_vocab.save(os.path.join(save_dir, "param_vocab.pkl"))
    with open(os.path.join(save_dir, "atom_registry.pkl"), 'wb') as f:
        pickle.dump(atom_registry, f)
    print(f"  Vocabularies saved to {save_dir}/")

    return opc_vocab, param_vocab, atom_registry


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import Config
    cfg = Config()
    build_all_vocabs(cfg)
