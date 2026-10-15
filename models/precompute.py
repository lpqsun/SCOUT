#!/usr/bin/env python3
"""
Precompute the 4 statistical features for every trace.

Features (Section 4.5.1 of the paper):
  1. jump_flag       — 1 if opcode is JUMP or JUMPI, else 0
  2. gas_cost        — normalized base gas cost (0..1)
  3. stack_depth_delta — d_i - d_{i-1}  (from raw trace's Depth column)
  4. storage_flag    — 1 if opcode is SLOAD or SSTORE, else 0

Run once before training; stores {trace}_stats.npy alongside each param_only.txt.
"""

import os
import sys
import json
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==============================================================================
# Opcode base gas costs  (from Ethereum Yellow Paper, Appendix G)
# ==============================================================================
OPCODE_BASE_GAS = {
    "STOP": 0, "ADD": 3, "MUL": 5, "SUB": 3, "DIV": 5, "SDIV": 5,
    "MOD": 5, "SMOD": 5, "ADDMOD": 8, "MULMOD": 8, "EXP": 50,
    "SIGNEXTEND": 5,
    "LT": 3, "GT": 3, "SLT": 3, "SGT": 3, "EQ": 3, "ISZERO": 3,
    "AND": 3, "OR": 3, "XOR": 3, "NOT": 3, "BYTE": 3,
    "SHL": 3, "SHR": 3, "SAR": 3,
    "SHA3": 30, "KECCAK256": 30,
    "ADDRESS": 2, "BALANCE": 700, "ORIGIN": 2, "CALLER": 2,
    "CALLVALUE": 2, "CALLDATALOAD": 3, "CALLDATASIZE": 2,
    "CALLDATACOPY": 3, "CODESIZE": 2, "CODECOPY": 3,
    "GASPRICE": 2, "EXTCODESIZE": 700, "EXTCODECOPY": 700,
    "RETURNDATASIZE": 2, "RETURNDATACOPY": 3,
    "EXTCODEHASH": 700, "BLOCKHASH": 20,
    "COINBASE": 2, "TIMESTAMP": 2, "NUMBER": 2, "PREVRANDAO": 2,
    "DIFFICULTY": 2, "GASLIMIT": 2, "CHAINID": 2,
    "SELFBALANCE": 5, "BASEFEE": 2,
    "POP": 2, "MLOAD": 3, "MSTORE": 3, "MSTORE8": 3,
    "SLOAD": 800, "SSTORE": 20000,
    "JUMP": 8, "JUMPI": 10, "PC": 2, "MSIZE": 2, "GAS": 2,
    "JUMPDEST": 1,
    "LOG0": 375, "LOG1": 750, "LOG2": 1125, "LOG3": 1500, "LOG4": 1875,
    "CREATE": 32000, "CALL": 700, "CALLCODE": 700, "RETURN": 0,
    "DELEGATECALL": 700, "CREATE2": 32000, "STATICCALL": 700,
    "REVERT": 0, "INVALID": 0, "SELFDESTRUCT": 5000,
    # Cancun
    "PUSH0": 2, "TLOAD": 100, "TSTORE": 100, "MCOPY": 3,
    # PUSHn  (base gas = 3)
    **{f"PUSH{i}": 3 for i in range(1, 33)},
    # DUPn, SWAPn  (base gas = 3)
    **{f"DUP{i}": 3 for i in range(1, 17)},
    **{f"SWAP{i}": 3 for i in range(1, 17)},
}

# Max gas cost for normalization
MAX_OPCODE_GAS = 32000  # CREATE / CREATE2

# ==============================================================================
# Precompute helpers
# ==============================================================================

def parse_trace_depths(trace_filepath: str):
    """Parse depth column from the original opcode trace .txt file."""
    depths = []
    with open(trace_filepath, 'r', encoding='utf-8', errors='replace') as f:
        in_trace = False
        for line in f:
            line = line.strip()
            if line.startswith('PC') and 'Opcode' in line:
                in_trace = True
                continue
            if line.startswith('---'):
                continue
            if not line:
                if in_trace:
                    break
                continue
            if in_trace:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        depths.append(int(parts[2]))
                    except ValueError:
                        break
    return depths


def precompute_trace(json_path: str, trace_dir: str) -> np.ndarray:
    """
    Compute (L, 4) statistical feature array for one trace.

    Args:
        json_path: path to {trace}_param_features.json
        trace_dir: path to the raw opcode trace directory

    Returns:
        np.ndarray of shape (L, 4), dtype float32
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    steps = data.get("steps", [])
    L = len(steps)
    if L == 0:
        return np.zeros((0, 4), dtype=np.float32)

    stats = np.zeros((L, 4), dtype=np.float32)

    # Extract opcode names
    opcodes = [s["opcode"] for s in steps]

    # 1. Jump flag
    jump_set = {"JUMP", "JUMPI"}
    stats[:, 0] = np.array([1.0 if op in jump_set else 0.0 for op in opcodes],
                           dtype=np.float32)

    # 2. Gas cost (normalized to [0, 1])
    gas_costs = np.array([OPCODE_BASE_GAS.get(op, 3) for op in opcodes],
                         dtype=np.float32)
    stats[:, 1] = gas_costs / MAX_OPCODE_GAS

    # 3. Stack depth delta
    basename = data.get("source", os.path.basename(json_path).replace("_param_features.json", ""))
    # Find the raw trace file: try common naming patterns
    trace_file = None
    for ext in [".txt", ""]:
        candidate = os.path.join(trace_dir, basename + ext)
        if os.path.exists(candidate):
            trace_file = candidate
            break
    # Also try alternative naming patterns
    if trace_file is None:
        for fname in os.listdir(trace_dir):
            if fname.startswith(basename) and fname.endswith(".txt") and ":Zone" not in fname:
                trace_file = os.path.join(trace_dir, fname)
                break

    if trace_file and os.path.exists(trace_file):
        depths = parse_trace_depths(trace_file)
        if len(depths) >= L:
            depths = depths[:L]
        elif len(depths) < L:
            depths = depths + [depths[-1]] * (L - len(depths)) if depths else [1] * L
        depth_arr = np.array(depths, dtype=np.float32)
        delta = np.zeros(L, dtype=np.float32)
        delta[0] = 0.0
        delta[1:] = depth_arr[1:] - depth_arr[:-1]
        stats[:, 2] = delta
    else:
        # Fallback: all zeros (no depth info available)
        stats[:, 2] = 0.0

    # 4. Storage access flag (includes transient storage from Cancun)
    storage_set = {"SLOAD", "SSTORE", "TLOAD", "TSTORE"}
    stats[:, 3] = np.array([1.0 if op in storage_set else 0.0 for op in opcodes],
                           dtype=np.float32)

    return stats


def precompute_all(config):
    """Precompute statistical features for all traces in all categories."""
    base_dir = config.DATA_ROOT

    total_traces = 0
    for category in config.CATEGORIES:
        param_dir = os.path.join(base_dir, category + config.PARAM_DIR_SUFFIX)
        trace_dir = os.path.join(base_dir, category)
        if not os.path.isdir(param_dir):
            print(f"  Skipping {category}: directory not found")
            continue

        json_files = sorted([
            f for f in os.listdir(param_dir)
            if f.endswith("_param_features.json")
        ])

        print(f"\n  {category}: {len(json_files)} traces")
        for i, fname in enumerate(json_files):
            json_path = os.path.join(param_dir, fname)
            try:
                stats = precompute_trace(json_path, trace_dir)
                out_path = json_path.replace("_param_features.json", "_stats.npy")
                np.save(out_path, stats)
                total_traces += 1
            except Exception as e:
                print(f"    ERROR {fname}: {e}")

            if (i + 1) % 500 == 0:
                print(f"    [{i+1}/{len(json_files)}] ...")

    # Compute global gas normalization
    print(f"\n  Total traces processed: {total_traces}")
    print(f"  Stats saved alongside param_features.json files")


if __name__ == "__main__":
    from config import Config
    cfg = Config()
    precompute_all(cfg)
