#!/usr/bin/env python3
"""
Batch generate parameter features + stats for normal_traces.
Produces the same 5-file output format as the vulnerability _ParamFeatures dirs.

Output per trace:
  {name}_param_features.json   — full features (opcode + param type + dependencies)
  {name}_param_only.txt        — parameter-only sequence (for CBOW training)
  {name}_dep_matrix.json       — sparse data dependency graph (for GCN)
  {name}_param_features.txt    — human-readable annotated trace
  {name}_stats.npy             — 4 statistical features (jump/gas/depth/storage)

Usage:
  python generate_normal_param_features.py
"""

import os, sys, json, time
import numpy as np

# Import from parent directory and models/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
from param_feature_simulator import simulate_parameter_features, parse_trace_file, format_parameter_sequence
from precompute import OPCODE_BASE_GAS, MAX_OPCODE_GAS

# ==============================================================================
# Paths
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NORM_TRACES_DIR = os.path.join(BASE_DIR, "normal_traces")
OUTPUT_DIR = os.path.join(BASE_DIR, "normal_traces_ParamFeatures")


# ==============================================================================
# Stats precomputation (inline from precompute.py)
# ==============================================================================
def compute_stats(ops, pcs, depths, trace_dir, basename):
    """Compute (L, 4) statistical features for a trace."""
    L = len(ops)
    if L == 0:
        return np.zeros((0, 4), dtype=np.float32)

    stats = np.zeros((L, 4), dtype=np.float32)

    # 1. Jump flag (JUMP / JUMPI)
    jump_set = {"JUMP", "JUMPI"}
    stats[:, 0] = np.array([1.0 if op in jump_set else 0.0 for op in ops], dtype=np.float32)

    # 2. Gas cost (normalized to [0, 1])
    gas_costs = np.array([OPCODE_BASE_GAS.get(op, 3) for op in ops], dtype=np.float32)
    stats[:, 1] = gas_costs / MAX_OPCODE_GAS

    # 3. Stack depth delta (from provided depths)
    depth_arr = np.array(depths, dtype=np.float32)
    delta = np.zeros(L, dtype=np.float32)
    delta[0] = 0.0
    if L > 1:
        delta[1:] = depth_arr[1:] - depth_arr[:-1]
    stats[:, 2] = delta

    # 4. Storage access flag (SLOAD / SSTORE / TLOAD / TSTORE)
    storage_set = {"SLOAD", "SSTORE", "TLOAD", "TSTORE"}
    stats[:, 3] = np.array([1.0 if op in storage_set else 0.0 for op in ops], dtype=np.float32)

    return stats


# ==============================================================================
# Main batch processor
# ==============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trace_files = sorted([f for f in os.listdir(NORM_TRACES_DIR)
                         if f.endswith('.txt') and not f.endswith('_param_features.txt')
                         and not f.endswith('_param_only.txt')])

    print(f"Generating param features for {len(trace_files)} normal traces...")
    print(f"  Input:  {NORM_TRACES_DIR}/")
    print(f"  Output: {OUTPUT_DIR}/")
    print()

    success = 0
    errors = 0
    total_ops = 0
    total_edges = 0
    t_start = time.time()

    for idx, filename in enumerate(trace_files):
        filepath = os.path.join(NORM_TRACES_DIR, filename)
        basename = filename.replace('.txt', '')

        try:
            # Step 1: Parse trace
            ops, pcs, depths, metadata = parse_trace_file(filepath)
            n = len(ops)
            if n == 0:
                raise ValueError("empty trace")

            # Step 2: Simulate parameter features (Algorithm 1)
            P, M = simulate_parameter_features(ops, depths, pcs, verbose=False)
            edge_count = sum(len(edges) for edges in M)

            # Step 3: Save _param_features.txt (human-readable)
            seq_output = format_parameter_sequence(ops, pcs, P, metadata=metadata,
                                                    include_dep=True, M=M)
            with open(os.path.join(OUTPUT_DIR, f"{basename}_param_features.txt"),
                     'w', encoding='utf-8') as f:
                f.write(f"# Parameter Feature Sequence for {basename}\n")
                f.write(f"# Total steps: {n}\n")
                if metadata:
                    for key, value in metadata.items():
                        f.write(f"#   {key}: {value}\n")
                f.write(f"#\n\n")
                f.write(seq_output)

            # Step 4: Save _param_only.txt (for CBOW training)
            with open(os.path.join(OUTPUT_DIR, f"{basename}_param_only.txt"),
                     'w', encoding='utf-8') as f:
                f.write("\n".join(P))

            # Step 5: Save _dep_matrix.json (sparse, for GCN)
            M_serializable = {str(k): v for k, v in enumerate(M) if v}
            with open(os.path.join(OUTPUT_DIR, f"{basename}_dep_matrix.json"),
                     'w', encoding='utf-8') as f:
                json.dump(M_serializable, f, indent=2)

            # Step 6: Save _param_features.json (full structured data)
            json_data = {
                "source": basename,
                "metadata": metadata,
                "total_steps": n,
                "steps": [
                    {"index": i, "pc": pcs[i], "opcode": ops[i],
                     "param_type": P[i], "dependencies": M[i] if M else []}
                    for i in range(n)
                ],
                "statistics": {
                    "null_count": sum(1 for p in P if p == "NULL"),
                    "single_type_count": sum(1 for p in P if p != "NULL" and not p.startswith("(")),
                    "multi_type_count": sum(1 for p in P if p.startswith("(")),
                    "dependency_edges": edge_count,
                },
            }
            with open(os.path.join(OUTPUT_DIR, f"{basename}_param_features.json"),
                     'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)

            # Step 7: Compute and save _stats.npy
            stats = compute_stats(ops, pcs, depths, NORM_TRACES_DIR, basename)
            np.save(os.path.join(OUTPUT_DIR, f"{basename}_stats.npy"), stats)

            success += 1
            total_ops += n
            total_edges += edge_count

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR [{filename[:60]}]: {e}")

        # Progress
        if (idx + 1) % 1000 == 0 or (idx + 1) == len(trace_files):
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{idx+1}/{len(trace_files)}] {rate:.1f} traces/s, "
                  f"{total_ops:,} ops, {total_edges:,} edges")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  COMPLETE: {success} ok, {errors} errors in {elapsed:.1f}s")
    print(f"  Total opcodes: {total_ops:,}")
    print(f"  Total dep edges: {total_edges:,}")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"  Files per trace: 5 (param_features.txt, param_only.txt, "
          f"dep_matrix.json, param_features.json, stats.npy)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
