#!/usr/bin/env python3
"""
Algorithm 1: Parameter Feature Sequence Simulation
From: Smart Contract Vulnerability Detection Method (Section 4.3.2)

Converts EVM opcode trace sequences into symbolic parameter feature sequences
by simulating stack behavior and tracking data dependencies.
"""

import os
import re
import json
from collections import defaultdict

# =============================================================================
# Table II: Parameter Type Mapping (参数操作码 → 符号类型)
# =============================================================================
TYPE_MAP = {
    # PUSHn → VALUE
    **{f"PUSH{i}": "VALUE" for i in range(1, 33)},
    "PUSH0": "VALUE",  # Cancun opcode

    # Account-related
    "ADDRESS":      "ACCOUNTADDRESS",
    "ORIGIN":       "ORIGINADDRESS",
    "CALLER":       "CALLERADDRESS",
    "COINBASE":     "COINBASEADDRESS",
    "SELFBALANCE":  "ACCOUNTBALANCE",

    # Value-related
    "CALLVALUE":    "ETHVALUE",

    # Environment-related
    "CALLDATASIZE":  "INPUTDATASIZE",
    "RETURNDATASIZE": "OUTPUTDATASIZE",
    "TIMESTAMP":     "BLOCKTIMESTAMP",
    "NUMBER":        "BLOCKNUMBER",
    "PREVRANDAO":    "BLOCKRANDAO",
    "DIFFICULTY":    "BLOCKDIFFICULTY",
    "CHAINID":       "ETHCHAINID",
    "PC":            "COUNTER",

    # Gas-related
    "GASLIMIT":  "BLOCKGASLIMIT",
    "GAS":       "GASCOST",
    "GASPRICE":  "GASPRICE",
    "BASEFEE":   "BASECOST",

    # Memory/Storage-related
    "CODESIZE":  "RUNCODESIZE",
    "MSIZE":     "MEMORYSIZE",
}

# =============================================================================
# Control Opcodes (控制操作码) — do not manipulate stack, no external data
# =============================================================================
CONTROL_OPCODES = {
    "STOP", "INVALID", "JUMPDEST", "RETURN", "REVERT",
    "SELFDESTRUCT",  # legacy
}

# =============================================================================
# "Other" Opcodes: stack behavior (pops, pushes)
# pop_behavior[opcode] = (num_pops, num_pushes)
# =============================================================================
POP_BEHAVIOR = {}

# Arithmetic: (2 pops, 1 push)
for op in ["ADD", "SUB", "MUL", "DIV", "SDIV", "MOD", "SMOD",
           "EXP", "SIGNEXTEND", "LT", "GT", "SLT", "SGT",
           "EQ", "AND", "OR", "XOR", "BYTE", "SHL", "SHR", "SAR"]:
    POP_BEHAVIOR[op] = (2, 1)

# Unary arithmetic: (1 pop, 1 push)
for op in ["ISZERO", "NOT"]:
    POP_BEHAVIOR[op] = (1, 1)

# Ternary: (3 pops, 1 push)
for op in ["ADDMOD", "MULMOD"]:
    POP_BEHAVIOR[op] = (3, 1)

# Stack manipulation
POP_BEHAVIOR["POP"] = (1, 0)

# Memory
POP_BEHAVIOR["MLOAD"] = (1, 1)
POP_BEHAVIOR["MSTORE"] = (2, 0)
POP_BEHAVIOR["MSTORE8"] = (2, 0)

# Storage
POP_BEHAVIOR["SLOAD"] = (1, 1)
POP_BEHAVIOR["SSTORE"] = (2, 0)

# Control flow
POP_BEHAVIOR["JUMP"] = (1, 0)
POP_BEHAVIOR["JUMPI"] = (2, 0)

# Hashing
POP_BEHAVIOR["KECCAK256"] = (2, 1)
POP_BEHAVIOR["SHA3"] = (2, 1)  # alias

# Call data
POP_BEHAVIOR["CALLDATALOAD"] = (1, 1)
POP_BEHAVIOR["CALLDATACOPY"] = (3, 0)

# Return data
POP_BEHAVIOR["RETURNDATACOPY"] = (3, 0)

# Code
POP_BEHAVIOR["CODECOPY"] = (3, 0)
POP_BEHAVIOR["EXTCODESIZE"] = (1, 1)
POP_BEHAVIOR["EXTCODECOPY"] = (4, 0)
POP_BEHAVIOR["EXTCODEHASH"] = (1, 1)

# Block
POP_BEHAVIOR["BLOCKHASH"] = (1, 1)

# Balance
POP_BEHAVIOR["BALANCE"] = (1, 1)

# Logging
for log_i in range(5):
    POP_BEHAVIOR[f"LOG{log_i}"] = (2 + log_i, 0)

# Call-family opcodes
POP_BEHAVIOR["CALL"] = (7, 1)
POP_BEHAVIOR["CALLCODE"] = (7, 1)
POP_BEHAVIOR["DELEGATECALL"] = (6, 1)
POP_BEHAVIOR["STATICCALL"] = (6, 1)
POP_BEHAVIOR["CREATE"] = (3, 1)
POP_BEHAVIOR["CREATE2"] = (4, 1)

# Transient storage (Cancun)
POP_BEHAVIOR["TLOAD"] = (1, 1)
POP_BEHAVIOR["TSTORE"] = (2, 0)

# Memory copy (Cancun)
POP_BEHAVIOR["MCOPY"] = (3, 0)

# =============================================================================
# DUPn and SWAPn stack positions
# DUPn: duplicates stack[-(n)] (1-indexed: DUP1 = stack[-1], DUP2 = stack[-2])
# SWAPn: swaps stack[-1] with stack[-(n+1)]
# =============================================================================

def parse_opcode_trace(filepath):
    """
    Parse an opcode trace file.
    Returns:
        opcodes: list of opcode strings
        depths: list of call depths (integers)
        metadata: dict of metadata from the file footer
    """
    opcodes = []
    depths = []
    metadata = {}

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    lines = content.split('\n')

    in_trace = False
    for line in lines:
        line = line.strip()
        if not line:
            if in_trace:
                # Empty line after trace section → metadata follows
                break
            continue

        # Detect start of trace data
        if line.startswith('PC') and 'Opcode' in line:
            in_trace = True
            continue
        if line.startswith('---'):
            continue

        if in_trace:
            # Parse trace line: PC, Opcode, Depth (tab-separated or whitespace)
            parts = line.split()
            if len(parts) >= 3:
                try:
                    pc = int(parts[0])
                    opcode = parts[1]
                    depth = int(parts[2])
                    opcodes.append(opcode)
                    depths.append(depth)
                except (ValueError, IndexError):
                    # End of trace data
                    in_trace = False
                    continue
            elif len(parts) == 1 and parts[0].isdigit():
                # Possibly a PC-only line (continuation), skip
                continue
            else:
                in_trace = False
                continue

    # Parse metadata
    metadata_section = content.split('\n\n')
    if len(metadata_section) > 1:
        for part in metadata_section[1:]:
            for line in part.split('\n'):
                line = line.strip()
                if line.startswith('===') and line.endswith('==='):
                    key_value = line.strip('= ').split(':', 1)
                    if len(key_value) == 2:
                        key = key_value[0].strip()
                        value = key_value[1].strip()
                        metadata[key] = value
                    elif len(key_value) == 1:
                        # Handle "=== Source: ... ===" format
                        pass

    return opcodes, depths, metadata


def is_param_opcode(opcode):
    """Check if opcode is a parameter opcode (pushes external data)."""
    return opcode in TYPE_MAP


def is_control_opcode(opcode):
    """Check if opcode is a control opcode."""
    return opcode in CONTROL_OPCODES


def get_dup_index(opcode):
    """Return the stack index for DUPn (1-based from top)."""
    if opcode.startswith('DUP'):
        try:
            return int(opcode[3:])
        except ValueError:
            pass
    return None


def get_swap_index(opcode):
    """Return the stack index for SWAPn (1-based from top, swaps top with n+1)."""
    if opcode.startswith('SWAP'):
        try:
            return int(opcode[4:])
        except ValueError:
            pass
    return None


def algorithm1(opcodes, depths):
    """
    Algorithm 1: Parameter Feature Sequence Simulation.

    Input:
        opcodes: list of opcode strings O = [o_1, ..., o_n]
        depths: list of call stack frame depths D = [d_1, ..., d_n]

    Output:
        P: parameter feature sequence [p_1, ..., p_n]
        M: data dependency matrix (sparse representation as dict of sets)
           M[i] = set of step indices j where step j's output is used by step i
    """
    n = len(opcodes)

    # S: list of stack frames, each frame is a list of parameter types
    S = []

    # Dep: dict (frame_index, stack_position) → step_index
    # Tracks which step produced the value at each stack position
    Dep = {}

    P = [None] * n
    M = defaultdict(set)

    # Frame counter for unique identification
    frame_counter = [0]

    def frame_id():
        """Get a unique identifier for the current frame."""
        return len(S) - 1 if S else 0

    def record_dep(frame_idx, pos, step_idx):
        """Record dependency: stack position (frame_idx, pos) was produced by step_idx."""
        Dep[(frame_idx, pos)] = step_idx

    def lookup_dep(frame_idx, pos):
        """Look up which step produced the value at (frame_idx, pos)."""
        return Dep.get((frame_idx, pos))

    for i in range(n):
        o_i = opcodes[i]
        d_i = depths[i]

        # Step 4-6: Adjust stack frames to match depth
        while len(S) < d_i:
            S.append([])  # Push new empty stack frame
        while len(S) > d_i:
            # Pop last stack frame — remove its Dep entries
            popped_frame_idx = len(S) - 1
            S.pop()
            # Clean up Dep entries for the popped frame
            keys_to_remove = [k for k in Dep if k[0] == popped_frame_idx]
            for k in keys_to_remove:
                del Dep[k]

        if not S:
            # Should not happen with valid depth data, but safety
            S.append([])

        frame_idx = len(S) - 1
        frame = S[frame_idx]

        # Step 8-9: Control opcode
        if is_control_opcode(o_i):
            P[i] = "NULL"

        # Step 10-13: Parameter opcode
        elif is_param_opcode(o_i):
            p_i = TYPE_MAP[o_i]
            P[i] = p_i
            frame.append(p_i)
            pos = len(frame) - 1
            record_dep(frame_idx, pos, i)

        # DUPn: special handling
        elif get_dup_index(o_i) is not None:
            dup_n = get_dup_index(o_i)  # 1-based from top
            if len(frame) >= dup_n:
                # Read the type at stack[-(dup_n)]
                dup_type = frame[-dup_n]
                P[i] = dup_type
                # Push the duplicate
                frame.append(dup_type)
                pos = len(frame) - 1
                record_dep(frame_idx, pos, i)
                # Data dependency: new value depends on the source
                src_pos = len(frame) - 1 - dup_n
                src_step = lookup_dep(frame_idx, src_pos)
                if src_step is not None:
                    M[i].add(src_step)
            else:
                # Stack underflow — treat as NULL
                P[i] = "NULL"

        # SWAPn: special handling — swap without creating new values
        elif get_swap_index(o_i) is not None:
            swap_n = get_swap_index(o_i)
            if len(frame) >= swap_n + 1:
                pos1 = len(frame) - 1          # top of stack
                pos2 = len(frame) - (swap_n + 1)  # position swap_n below top

                type1 = frame[pos1]
                type2 = frame[pos2]

                # Build combined type for p_i
                combined = sorted(set([type1, type2]))
                P[i] = " ".join(combined) if combined else "NULL"

                # Swap the types in the frame
                frame[pos1], frame[pos2] = frame[pos2], frame[pos1]

                # Swap Dep entries
                step1 = lookup_dep(frame_idx, pos1)
                step2 = lookup_dep(frame_idx, pos2)
                if step1 is not None:
                    Dep[(frame_idx, pos2)] = step1
                if step2 is not None:
                    Dep[(frame_idx, pos1)] = step2
            else:
                P[i] = "NULL"

        # Step 14-25: Other opcode
        else:
            behavior = POP_BEHAVIOR.get(o_i)
            if behavior is None:
                # Unknown opcode — treat as no stack effect
                P[i] = "NULL"
                continue

            need_pops, num_pushes = behavior

            args = []
            popped_steps = []

            if need_pops > 0:
                if len(frame) >= need_pops:
                    # Pop from end of frame
                    for _ in range(need_pops):
                        val = frame.pop()
                        args.append(val)
                        # Look up which step produced this value
                        pos = len(frame)  # position after pop
                        src_step = lookup_dep(frame_idx, pos)
                        if src_step is not None:
                            popped_steps.append(src_step)
                        # Remove old Dep entry
                        if (frame_idx, pos) in Dep:
                            del Dep[(frame_idx, pos)]
                else:
                    # Stack underflow — use what's available
                    while frame:
                        val = frame.pop()
                        args.append(val)
                    P[i] = "NULL"
                    continue

            # Reverse args (they were popped in reverse order)
            args.reverse()

            # Step 17: Sort lexicographically, deduplicate
            combined = sorted(set(args))
            if combined:
                P[i] = " ".join(combined)
            else:
                P[i] = "NULL"

            # Step 19-21: Record data dependencies
            for src_step in popped_steps:
                M[i].add(src_step)

            # Step 22-24: Push result if applicable
            if num_pushes > 0:
                for _ in range(num_pushes):
                    if combined:
                        frame.append(P[i])
                    else:
                        frame.append("NULL")
                    pos = len(frame) - 1
                    record_dep(frame_idx, pos, i)

    # Convert M to standard dict of lists for JSON serialization
    M_dict = {str(k): sorted(list(v)) for k, v in M.items()}

    return P, M_dict


def format_parameter_sequence(opcodes, depths, P, metadata, output_path):
    """
    Save the parameter feature sequence to a file in a readable format.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Step\tOpcode\t\tDepth\tParameter_Feature\n")
        f.write("----\t------\t\t-----\t-----------------\n")

        for i in range(len(opcodes)):
            f.write(f"{i+1:4d}\t{opcodes[i]:<12s}\t{depths[i]:5d}\t{P[i]}\n")

        f.write("\n")

        # Write original metadata
        if metadata:
            for key, value in metadata.items():
                f.write(f"=== {key}: {value} ===\n")


def save_dependency_matrix(M, output_path):
    """
    Save the data dependency matrix in sparse JSON format.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(M, f, indent=2)


import sys
import time

# =============================================================================
# Subdirectories to process
# =============================================================================
SUBDIRS = [
    "balance_check",
    "block_number_dependency",
    "integer_overflow",
    "reentrancy",
    "timestamp_dependency",
    "unchecked_external_call",
]


def process_directory(input_dir, output_dir):
    """
    Process all opcode trace files in a directory.
    Returns (success_count, error_count, total_opcodes, total_edges).
    """
    if not os.path.isdir(input_dir):
        print(f"  Directory not found: {input_dir}")
        return 0, 0, 0, 0

    trace_files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith('.txt') and not f.endswith(':Zone.Identifier')
    ])

    if not trace_files:
        print(f"  No trace files found in {input_dir}")
        return 0, 0, 0, 0

    os.makedirs(output_dir, exist_ok=True)

    success = 0
    errors = 0
    total_opcodes = 0
    total_edges = 0
    t_start = time.time()

    for idx, filename in enumerate(trace_files):
        filepath = os.path.join(input_dir, filename)

        try:
            opcodes, depths, metadata = parse_opcode_trace(filepath)

            if not opcodes:
                errors += 1
                continue

            # Run Algorithm 1
            P, M = algorithm1(opcodes, depths)

            total_opcodes += len(opcodes)
            total_edges += sum(len(v) for v in M.values())

            # Save parameter feature sequence
            base_name = filename.replace('.txt', '')
            param_output = os.path.join(output_dir, f"{base_name}_param_features.txt")
            format_parameter_sequence(opcodes, depths, P, metadata, param_output)

            # Save dependency matrix (sparse JSON)
            dep_output = os.path.join(output_dir, f"{base_name}_dep_matrix.json")
            save_dependency_matrix(M, dep_output)

            # Save compact P sequence only
            compact_output = os.path.join(output_dir, f"{base_name}_param_seq.txt")
            with open(compact_output, 'w', encoding='utf-8') as f:
                for i, p in enumerate(P):
                    f.write(f"{i+1}\t{p}\n")

            success += 1

            # Progress update every 500 files or at the end
            if (idx + 1) % 500 == 0 or (idx + 1) == len(trace_files):
                elapsed = time.time() - t_start
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                print(f"    [{idx+1}/{len(trace_files)}] "
                      f"{rate:.1f} files/s, "
                      f"ops: {total_opcodes}, edges: {total_edges}")

        except Exception as e:
            errors += 1
            if errors <= 5:  # Only print first 5 errors to avoid flooding
                print(f"  ERROR [{filename}]: {e}")
            continue

    elapsed = time.time() - t_start
    print(f"  Completed: {success} ok, {errors} errors in {elapsed:.1f}s "
          f"({success/elapsed:.1f} files/s)" if elapsed > 0 else "")
    return success, errors, total_opcodes, total_edges


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Optional: process TestVulnerability if argument given
    process_testvul = "--include-test" in sys.argv

    all_dirs = list(SUBDIRS)
    if process_testvul:
        all_dirs.insert(0, "TestVulnerability")

    grand_success = 0
    grand_errors = 0
    grand_opcodes = 0
    grand_edges = 0
    t0 = time.time()

    for subdir in all_dirs:
        input_dir = os.path.join(base_dir, subdir)
        output_dir = os.path.join(base_dir, f"{subdir}_ParamFeatures")

        print(f"\n{'='*60}")
        print(f"Processing: {subdir}/")
        print(f"  Input:  {input_dir}")
        print(f"  Output: {output_dir}")

        s, e, ops, edges = process_directory(input_dir, output_dir)
        grand_success += s
        grand_errors += e
        grand_opcodes += ops
        grand_edges += edges

    total_elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"ALL DONE. Total: {grand_success} files processed, "
          f"{grand_errors} errors")
    print(f"  Total opcodes: {grand_opcodes:,}")
    print(f"  Total dependency edges: {grand_edges:,}")
    print(f"  Time: {total_elapsed:.1f}s "
          f"({grand_success/total_elapsed:.1f} files/s)" if total_elapsed > 0 else "")


if __name__ == "__main__":
    main()
