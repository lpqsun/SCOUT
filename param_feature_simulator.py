#!/usr/bin/env python3
"""
Algorithm 1: Parameter Feature Sequence Simulation
Based on "Smart Contract Vulnerability Detection Method" paper, Section 4.3.

Converts EVM opcode execution traces into symbolic parameter feature sequences
by simulating the EVM stack's data flow, classifying each opcode's parameter
type according to its data origin.
"""

import os
import json
from typing import List, Tuple, Optional

# ==============================================================================
# Table II: Parameter Type Mapping (Complete)
# ==============================================================================

# Control opcodes → parameter type = NULL
CONTROL_OPCODES = {
    "STOP", "INVALID", "JUMPDEST", "RETURN", "REVERT",
}

# Parameter opcodes → symbolic parameter type (Table II)
PARAM_TYPE_MAP = {
    # PUSHn → VALUE
    **{f"PUSH{i}": "VALUE" for i in range(1, 33)},
    # Account-related
    "ADDRESS": "ACCOUNTADDRESS",
    "ORIGIN": "ORIGINADDRESS",
    "CALLER": "CALLERADDRESS",
    "COINBASE": "COINBASEADDRESS",
    # Value-related
    "CALLVALUE": "ETHVALUE",
    # Environment-related
    "CALLDATASIZE": "INPUTDATASIZE",
    "RETURNDATASIZE": "OUTPUTDATASIZE",
    "TIMESTAMP": "BLOCKTIMESTAMP",
    "NUMBER": "BLOCKNUMBER",
    "PREVRANDAO": "BLOCKRANDAO",
    "DIFFICULTY": "BLOCKDIFFICULTY",
    "CHAINID": "ETHCHAINID",
    "PC": "COUNTER",
    # Memory/Storage-related
    "CODESIZE": "RUNCODESIZE",
    "MSIZE": "MEMORYSIZE",
    # Gas-related
    "GASLIMIT": "BLOCKGASLIMIT",
    "GAS": "GASCOST",
    "GASPRICE": "GASPRICE",
    "SELFBALANCE": "ACCOUNTBALANCE",
    "BASEFEE": "BASECOST",
}

# Semantic groups (for reference)
SEMANTIC_GROUPS = {
    "BLOCKTIMESTAMP": "Environment",
    "BLOCKNUMBER": "Environment",
    "ETHCHAINID": "Environment",
    "INPUTDATASIZE": "Environment",
    "OUTPUTDATASIZE": "Environment",
    "BLOCKRANDAO": "Environment",
    "BLOCKDIFFICULTY": "Environment",
    "COUNTER": "Environment",
    "ACCOUNTADDRESS": "Account",
    "ORIGINADDRESS": "Account",
    "CALLERADDRESS": "Account",
    "COINBASEADDRESS": "Account",
    "ACCOUNTBALANCE": "Account",
    "VALUE": "Numeric",
    "ETHVALUE": "Numeric",
    "GASCOST": "Gas",
    "GASPRICE": "Gas",
    "BLOCKGASLIMIT": "Gas",
    "BASECOST": "Gas",
    "RUNCODESIZE": "Memory/Storage",
    "MEMORYSIZE": "Memory/Storage",
}

# ==============================================================================
# EVM Opcode Stack Behavior Database
# ==============================================================================

# (pop_count, push_count) for every EVM opcode
# Based on Ethereum Yellow Paper / evm.codes
OPCODE_STACK_BEHAVIOR = {
    # Stop/Arithmetic (0x00-0x0f)
    "STOP": (0, 0),
    "ADD": (2, 1),
    "MUL": (2, 1),
    "SUB": (2, 1),
    "DIV": (2, 1),
    "SDIV": (2, 1),
    "MOD": (2, 1),
    "SMOD": (2, 1),
    "ADDMOD": (3, 1),
    "MULMOD": (3, 1),
    "EXP": (2, 1),
    "SIGNEXTEND": (2, 1),

    # Comparison/Bitwise (0x10-0x1f)
    "LT": (2, 1),
    "GT": (2, 1),
    "SLT": (2, 1),
    "SGT": (2, 1),
    "EQ": (2, 1),
    "ISZERO": (1, 1),
    "AND": (2, 1),
    "OR": (2, 1),
    "XOR": (2, 1),
    "NOT": (1, 1),
    "BYTE": (2, 1),
    "SHL": (2, 1),
    "SHR": (2, 1),
    "SAR": (2, 1),

    # SHA3 (0x20)
    "SHA3": (2, 1),
    "KECCAK256": (2, 1),

    # Environment (0x30-0x3f)
    "ADDRESS": (0, 1),
    "BALANCE": (1, 1),
    "ORIGIN": (0, 1),
    "CALLER": (0, 1),
    "CALLVALUE": (0, 1),
    "CALLDATALOAD": (1, 1),
    "CALLDATASIZE": (0, 1),
    "CALLDATACOPY": (3, 0),
    "CODESIZE": (0, 1),
    "CODECOPY": (3, 0),
    "GASPRICE": (0, 1),
    "EXTCODESIZE": (1, 1),
    "EXTCODECOPY": (4, 0),
    "RETURNDATASIZE": (0, 1),
    "RETURNDATACOPY": (3, 0),
    "EXTCODEHASH": (1, 1),

    # Block (0x40-0x4f)
    "BLOCKHASH": (1, 1),
    "COINBASE": (0, 1),
    "TIMESTAMP": (0, 1),
    "NUMBER": (0, 1),
    "PREVRANDAO": (0, 1),
    "DIFFICULTY": (0, 1),
    "GASLIMIT": (0, 1),
    "CHAINID": (0, 1),
    "SELFBALANCE": (0, 1),
    "BASEFEE": (0, 1),

    # Stack/Memory/Storage (0x50-0x5f)
    "POP": (1, 0),
    "MLOAD": (1, 1),
    "MSTORE": (2, 0),
    "MSTORE8": (2, 0),
    "SLOAD": (1, 1),
    "SSTORE": (2, 0),
    "JUMP": (1, 0),
    "JUMPI": (2, 0),
    "PC": (0, 1),
    "MSIZE": (0, 1),
    "GAS": (0, 1),
    "JUMPDEST": (0, 0),

    # DUP (0x80-0x8f)
    **{f"DUP{i}": (0, 1) for i in range(1, 17)},

    # SWAP (0x90-0x9f)
    **{f"SWAP{i}": (0, 0) for i in range(1, 17)},

    # LOG (0xa0-0xa4)
    "LOG0": (2, 0),
    "LOG1": (3, 0),
    "LOG2": (4, 0),
    "LOG3": (5, 0),
    "LOG4": (6, 0),

    # System (0xf0-0xff)
    "CREATE": (3, 1),
    "CALL": (7, 1),
    "CALLCODE": (7, 1),
    "RETURN": (2, 0),
    "DELEGATECALL": (6, 1),
    "CREATE2": (4, 1),
    "STATICCALL": (6, 1),
    "REVERT": (2, 0),
    "INVALID": (0, 0),
    "SELFDESTRUCT": (1, 0),
}


def get_opcode_category(opcode: str) -> str:
    """Classify opcode into 'control', 'parameter', or 'other'."""
    if opcode in CONTROL_OPCODES:
        return "control"
    if opcode in PARAM_TYPE_MAP:
        return "parameter"
    return "other"


def get_stack_behavior(opcode: str) -> Tuple[int, int]:
    """Return (pop_count, push_count) for an opcode."""
    return OPCODE_STACK_BEHAVIOR.get(opcode, (0, 0))


# ==============================================================================
# Trace File Parser
# ==============================================================================

def parse_trace_file(filepath: str) -> Tuple[List[str], List[int], List[int], dict]:
    """
    Parse an opcode trace file.

    Returns:
        ops: list of opcode mnemonics
        pcs: list of program counters
        depths: list of stack frame depths
        metadata: dict of contract metadata from file footer
    """
    ops = []
    pcs = []
    depths = []
    metadata = {}

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    lines = content.split('\n')
    in_trace = False

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            if in_trace:
                # Empty line after trace section → remaining lines are metadata
                in_trace = False
            continue

        # Detect start of trace data
        if line_stripped.startswith('PC') and 'Opcode' in line_stripped:
            in_trace = True
            continue
        if line_stripped.startswith('---'):
            continue

        if in_trace:
            # Parse trace line: PC, Opcode, Depth
            parts = line_stripped.split()
            if len(parts) >= 3:
                try:
                    pc = int(parts[0])
                    opcode = parts[1]
                    depth = int(parts[2])
                    ops.append(opcode)
                    pcs.append(pc)
                    depths.append(depth)
                except (ValueError, IndexError):
                    # End of trace data
                    in_trace = False
                    continue
            elif len(parts) == 1 and parts[0].isdigit():
                # Possibly a PC-only continuation line, skip
                continue
            else:
                in_trace = False
                continue

    # Parse metadata from remaining content after trace
    # Metadata lines follow the format: === Key: Value === (or === Key Value ===)
    metadata_section = content.split('\n\n')
    for section in metadata_section[1:]:
        for line in section.split('\n'):
            line = line.strip()
            if line.startswith('===') and line.endswith('==='):
                inner = line.strip('= ').strip()
                # Try "Key: Value" format
                if ': ' in inner:
                    key, value = inner.split(': ', 1)
                    metadata[key.strip()] = value.strip()
                elif ':' in inner:
                    key, value = inner.split(':', 1)
                    metadata[key.strip()] = value.strip()
                else:
                    # "=== Source: ... ===" or "=== Key Value ===" format
                    parts = inner.split(None, 1)
                    if len(parts) == 2:
                        key = parts[0].rstrip(':')
                        metadata[key.strip()] = parts[1].strip()
                    elif len(parts) == 1:
                        metadata[parts[0].strip()] = ""

    return ops, pcs, depths, metadata


# ==============================================================================
# Algorithm 1: Parameter Feature Sequence Simulation
# ==============================================================================

def _format_param_type(atomic_types: frozenset) -> str:
    """Format a set of atomic parameter types into the canonical string form.

    Single type → just the type name.
    Multiple types → "(TYPE1 TYPE2 ...)" sorted lexicographically.
    Empty → "NULL"
    """
    if not atomic_types:
        return "NULL"
    sorted_types = sorted(atomic_types)
    if len(sorted_types) == 1:
        return sorted_types[0]
    return "(" + " ".join(sorted_types) + ")"


def simulate_parameter_features(
    ops: List[str],
    depths: List[int],
    pcs: List[int],
    verbose: bool = False,
) -> Tuple[List[str], List[List[int]]]:
    """
    Algorithm 1: Simulate EVM stack execution to derive symbolic parameter
    feature sequences and data dependency matrix.

    Each stack position stores a FROZENSET of atomic parameter types (not nested
    strings), so that later consumers correctly flatten and deduplicate.
    The parallel Producer stack stores the SET of ALL original-producing step
    indices for the types at that position.

    Args:
        ops: Opcode sequence O = [o_1, ..., o_n]
        depths: Stack frame depth sequence D = [d_1, ..., d_n]
        pcs: Program counters
        verbose: If True, print detailed simulation trace

    Returns:
        P: Parameter feature sequence [p_1, ..., p_n]
        M_sparse: Sparse dependency matrix as adjacency list M[i] = [j_1, j_2, ...]
    """
    n = len(ops)

    # Line 1：初始化空栈 S（每个栈帧是一个列表，元素为 frozenset of atomic types）
    S: List[List[frozenset]] = []

    # Parallel: each frame position tracks ALL original-producing step indices
    # (a frozenset of ints), so that flattening & transitive deps work correctly.
    Producers: List[List[frozenset]] = []

    # Line 2：Dep dict not strictly needed anymore — producer info lives in
    # Producers stack.  Kept for conceptual alignment with the paper.

    # Output parameter feature sequence
    P: List[str] = []

    # Sparse dependency matrix: adjacency list  M[i] = [j1, j2, ...]
    M: List[List[int]] = [[] for _ in range(n)]

    if verbose:
        print(f"{'Step':>4} {'PC':>5} {'Opcode':>12} {'Depth':>5} {'Frame':>50} {'P_i'}")
        print("-" * 130)

    for i in range(n):
        o_i = ops[i]
        depth = depths[i]

        # Lines 5-6：Adjust stack depth (add/remove frames)
        while len(S) < depth:
            S.append([])          # new empty frame
            Producers.append([])
        while len(S) > depth:
            S.pop()               # discard deepest frame
            Producers.pop()

        # Line 7：Get current frame (top of S)
        if not S:
            S.append([])
            Producers.append([])
        frame = S[-1]
        frame_producers = Producers[-1]
        frame_id = len(S) - 1

        # ── Line 8-9：Control opcodes ──
        if get_opcode_category(o_i) == "control":
            p_i = "NULL"

        # ── Line 10-13：Parameter opcodes ──
        elif get_opcode_category(o_i) == "parameter":
            atomic_type = PARAM_TYPE_MAP[o_i]
            p_i = atomic_type  # single atomic type → bare string
            frame.append(frozenset([atomic_type]))
            frame_producers.append(frozenset([i]))

        # ── Line 14-25：Other opcodes ──
        else:
            pop_count, push_count = get_stack_behavior(o_i)

            # -------- SWAPn: rearrange stack --------
            if o_i.startswith("SWAP"):
                n_swap = int(o_i[4:])          # SWAP1..SWAP16
                total = n_swap + 1              # items involved

                popped_types: List[frozenset] = []
                popped_prods: List[frozenset] = []
                if len(frame) >= total:
                    for _ in range(total):
                        popped_types.append(frame.pop())
                        popped_prods.append(frame_producers.pop())
                    popped_types.reverse()
                    popped_prods.reverse()

                # p_i = union of all involved atomic types
                all_atomic = frozenset.union(*popped_types) if popped_types else frozenset()
                p_i = _format_param_type(all_atomic)

                # Dependencies → all original producers
                all_prods = frozenset.union(*popped_prods) if popped_prods else frozenset()
                for j in sorted(all_prods):
                    M[i].append(j)

                # Push back in swapped order
                idx_top = len(popped_types) - 1
                idx_nth = len(popped_types) - 1 - n_swap
                popped_types[idx_top], popped_types[idx_nth] = \
                    popped_types[idx_nth], popped_types[idx_top]
                popped_prods[idx_top], popped_prods[idx_nth] = \
                    popped_prods[idx_nth], popped_prods[idx_top]

                for t, pr in zip(popped_types, popped_prods):
                    frame.append(t)
                    frame_producers.append(pr)

                P.append(p_i)
                continue

            # -------- DUPn: duplicate nth stack item --------
            if o_i.startswith("DUP"):
                dup_n = int(o_i[3:])             # DUP1..DUP16
                idx = len(frame) - dup_n          # 0-indexed from bottom
                if 0 <= idx < len(frame):
                    copied = frame[idx]
                    copied_prod = frame_producers[idx]
                    p_i = _format_param_type(copied)

                    frame.append(copied)
                    frame_producers.append(copied_prod)

                    for j in sorted(copied_prod):
                        M[i].append(j)
                else:
                    p_i = "NULL"
                    if push_count > 0:
                        frame.append(frozenset(["VALUE"]))
                        frame_producers.append(frozenset([i]))

                P.append(p_i)
                continue

            # -------- Normal other opcode: pop args, optionally push result --------
            need = pop_count
            popped_types: List[frozenset] = []
            popped_prods: List[frozenset] = []

            actual_pop = min(need, len(frame))
            for _ in range(actual_pop):
                popped_types.append(frame.pop())
                popped_prods.append(frame_producers.pop())

            # Handle stack underflow with conservative default
            if actual_pop < need:
                for _ in range(need - actual_pop):
                    popped_types.append(frozenset(["VALUE"]))
                    popped_prods.append(frozenset())

            # Line 17：atomic union → sort lexicographically, deduplicate
            all_atomic = frozenset.union(*popped_types) if popped_types else frozenset()
            p_i = _format_param_type(all_atomic)

            # Lines 19-21：Record dependencies (ALL original producers)
            all_prods = frozenset.union(*popped_prods) if popped_prods else frozenset()
            for j in sorted(all_prods):
                M[i].append(j)

            # Lines 22-25：Push result if applicable
            if push_count > 0:
                frame.append(all_atomic)
                frame_producers.append(frozenset([i]))

        P.append(p_i)

        if verbose:
            display_frame = "[" + ", ".join(
                _format_param_type(ts) for ts in frame
            ) + "]" if frame else "[]"
            print(f"{i:>4} {pcs[i]:>5} {o_i:>12} {depth:>5} {display_frame[:60]:>60} {p_i}")

    return P, M


# ==============================================================================
# Output Formatting & Saving
# ==============================================================================

def format_parameter_sequence(
    ops: List[str],
    pcs: List[int],
    P: List[str],
    metadata: Optional[dict] = None,
    include_dep: bool = False,
    M: Optional[List[List[int]]] = None,
) -> str:
    """Format the parameter feature sequence for output."""
    lines = []
    # Header with metadata
    if metadata:
        for key, value in metadata.items():
            lines.append(f"=== {key}: {value} ===")
        lines.append("")
    lines.append(f"{'Step':>5} {'PC':>6} {'Opcode':>15} {'ParamType'}")
    lines.append("-" * 60)
    for i, (op, pc, p) in enumerate(zip(ops, pcs, P)):
        dep_str = ""
        if include_dep and M and M[i]:
            dep_str = f"  ← steps: {M[i]}"
        lines.append(f"{i:>5} {pc:>6} {op:>15} {p}{dep_str}")
    return "\n".join(lines)


def process_trace_file(filepath: str, output_dir: str, verbose: bool = False):
    """Process a single trace file: parse, simulate, save results."""
    basename = os.path.splitext(os.path.basename(filepath))[0]
    print(f"\n{'='*60}")
    print(f"Processing: {basename}")
    print(f"{'='*60}")

    # Parse
    ops, pcs, depths, metadata = parse_trace_file(filepath)
    print(f"  Parsed {len(ops)} opcodes")
    if metadata:
        contract = metadata.get("Contract", metadata.get("Source", "unknown"))
        vuln = metadata.get("Vulnerability", "unknown")
        verdict = metadata.get("Verdict", "unknown")
        print(f"  Contract: {contract} | Vulnerability: {vuln} | Verdict: {verdict}")

    if verbose:
        print(f"  First 5 opcodes: {ops[:5]}")
        print(f"  Depths: min={min(depths)}, max={max(depths)}")

    # Simulate
    P, M = simulate_parameter_features(ops, depths, pcs, verbose=verbose)

    # Statistics
    unique_params = sorted(set(p for p in P if p != "NULL"))
    null_count = sum(1 for p in P if p == "NULL")
    multi_count = sum(1 for p in P if p.startswith("("))
    single_count = sum(1 for p in P if p != "NULL" and not p.startswith("("))
    dep_edges = sum(len(edges) for edges in M)

    print(f"  Parameter types found: {unique_params}")
    print(f"  NULL: {null_count}, Single: {single_count}, Multi-type: {multi_count}")
    print(f"  Dependency edges: {dep_edges}")

    # Save full parameter sequence
    seq_output = format_parameter_sequence(ops, pcs, P, metadata=metadata, include_dep=True, M=M)
    output_path = os.path.join(output_dir, f"{basename}_param_features.txt")
    with open(output_path, 'w') as f:
        f.write(f"# Parameter Feature Sequence for {basename}\n")
        f.write(f"# Total steps: {len(ops)}\n")
        f.write(f"# Unique param types: {len(unique_params)}\n")
        f.write(f"# NULL count: {null_count}\n")
        f.write(f"# Single-type count: {single_count}\n")
        f.write(f"# Multi-type count: {multi_count}\n")
        f.write(f"# Dependency edges: {dep_edges}\n")
        if metadata:
            f.write(f"#\n")
            for key, value in metadata.items():
                f.write(f"#   {key}: {value}\n")
        f.write(f"#\n")
        f.write(f"# Legend:\n")
        f.write(f"#   NULL = control opcode (STOP/JUMPDEST/RETURN/REVERT/INVALID)\n")
        f.write(f"#   VALUE = PUSHn immediate value\n")
        f.write(f"#   (TYPE1 TYPE2) = combined types from popped args\n")
        f.write(f"#\n\n")
        f.write(seq_output)

    print(f"  Saved: {output_path}")

    # Save parameter-only sequence (for model input)
    param_only_path = os.path.join(output_dir, f"{basename}_param_only.txt")
    with open(param_only_path, 'w') as f:
        f.write("\n".join(P))
    print(f"  Saved: {param_only_path}")

    # Save JSON format with full data
    json_path = os.path.join(output_dir, f"{basename}_param_features.json")
    json_data = {
        "source": basename,
        "metadata": metadata,
        "total_steps": len(ops),
        "steps": [
            {
                "index": i,
                "pc": pcs[i],
                "opcode": ops[i],
                "param_type": P[i],
                "dependencies": M[i] if M else [],
            }
            for i in range(len(ops))
        ],
        "statistics": {
            "unique_param_types": unique_params,
            "null_count": null_count,
            "single_type_count": single_count,
            "multi_type_count": multi_count,
            "dependency_edges": dep_edges,
        },
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  Saved: {json_path}")

    return {
        "basename": basename,
        "metadata": metadata,
        "total_steps": len(ops),
        "unique_params": unique_params,
        "null_count": null_count,
        "single_count": single_count,
        "multi_count": multi_count,
        "dep_edges": dep_edges,
    }


# ==============================================================================
# Batch Directory Processing
# ==============================================================================

# Directories to process (same structure as algorithm1_simulation.py)
SUBDIRS = [
    "balance_check",
    "block_number_dependency",
    "integer_overflow",
    "reentrancy",
    "timestamp_dependency",
    "unchecked_external_call",
]


def process_directory(input_dir: str, output_dir: str) -> dict:
    """
    Process all opcode trace files in a directory.
    Returns summary dict with counts and aggregate statistics.
    """
    if not os.path.isdir(input_dir):
        print(f"  Directory not found: {input_dir}")
        return {"success": 0, "errors": 0, "total_opcodes": 0, "total_edges": 0}

    trace_files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith('.txt') and ':Zone.Identifier' not in f
    ])

    if not trace_files:
        print(f"  No trace files found in {input_dir}")
        return {"success": 0, "errors": 0, "total_opcodes": 0, "total_edges": 0}

    os.makedirs(output_dir, exist_ok=True)

    success = 0
    errors = 0
    total_opcodes = 0
    total_edges = 0
    unique_param_types_all = set()
    t_start = time.time()

    for idx, filename in enumerate(trace_files):
        filepath = os.path.join(input_dir, filename)

        try:
            ops, pcs, depths, metadata = parse_trace_file(filepath)
            if not ops:
                errors += 1
                continue

            P, M = simulate_parameter_features(ops, depths, pcs, verbose=False)

            n = len(ops)
            total_opcodes += n
            edge_count = sum(len(edges) for edges in M)
            total_edges += edge_count

            # Track unique parameter types across all files
            for p in P:
                if p != "NULL":
                    unique_param_types_all.add(p)

            basename = filename.replace('.txt', '')

            # Save parameter feature sequence (full)
            seq_output = format_parameter_sequence(ops, pcs, P, metadata=metadata, include_dep=True, M=M)
            param_path = os.path.join(output_dir, f"{basename}_param_features.txt")
            with open(param_path, 'w', encoding='utf-8') as f:
                f.write(f"# Parameter Feature Sequence for {basename}\n")
                f.write(f"# Total steps: {n}\n")
                if metadata:
                    f.write(f"#\n")
                    for key, value in metadata.items():
                        f.write(f"#   {key}: {value}\n")
                f.write(f"#\n")
                f.write(f"# Legend: NULL=control opcode, VALUE=PUSHn, (X Y)=combined types\n")
                f.write(f"#\n\n")
                f.write(seq_output)

            # Save parameter-only sequence (for model training)
            compact_path = os.path.join(output_dir, f"{basename}_param_only.txt")
            with open(compact_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(P))

            # Save dependency matrix (sparse JSON)
            dep_path = os.path.join(output_dir, f"{basename}_dep_matrix.json")
            M_serializable = {str(k): v for k, v in enumerate(M) if v}
            with open(dep_path, 'w', encoding='utf-8') as f:
                json.dump(M_serializable, f, indent=2)

            # Save full JSON
            json_path = os.path.join(output_dir, f"{basename}_param_features.json")
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
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)

            success += 1

            # Progress every 500 files or at the end
            if (idx + 1) % 500 == 0 or (idx + 1) == len(trace_files):
                elapsed = time.time() - t_start
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                print(f"    [{idx+1}/{len(trace_files)}] "
                      f"{rate:.1f} files/s, "
                      f"ops: {total_opcodes:,}, edges: {total_edges:,}")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERROR [{filename}]: {e}")
            continue

    elapsed = time.time() - t_start
    print(f"  Completed: {success} ok, {errors} errors in {elapsed:.1f}s "
          f"({success/elapsed:.1f} files/s)" if elapsed > 0 else "")

    return {
        "success": success,
        "errors": errors,
        "total_opcodes": total_opcodes,
        "total_edges": total_edges,
        "unique_param_types": sorted(unique_param_types_all),
    }


# ==============================================================================
# Main
# ==============================================================================

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    grand_success = 0
    grand_errors = 0
    grand_opcodes = 0
    grand_edges = 0
    all_summaries = {}
    t0 = time.time()

    for subdir in SUBDIRS:
        input_dir = os.path.join(base_dir, subdir)
        output_dir = os.path.join(base_dir, f"{subdir}_ParamFeatures")

        print(f"\n{'='*60}")
        print(f"Processing: {subdir}/")
        print(f"  Input:  {input_dir}")
        print(f"  Output: {output_dir}")

        summary = process_directory(input_dir, output_dir)
        all_summaries[subdir] = summary
        grand_success += summary["success"]
        grand_errors += summary["errors"]
        grand_opcodes += summary["total_opcodes"]
        grand_edges += summary["total_edges"]

    total_elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"ALL DONE. Total: {grand_success:,} files processed, "
          f"{grand_errors:,} errors")
    print(f"  Total opcodes: {grand_opcodes:,}")
    print(f"  Total dependency edges: {grand_edges:,}")
    print(f"  Time: {total_elapsed:.1f}s "
          f"({grand_success/total_elapsed:.1f} files/s)" if total_elapsed > 0 else "")

    # Print per-directory summary
    print(f"\n{'='*60}")
    print(f"PER-DIRECTORY SUMMARY")
    print(f"{'='*60}")
    for subdir in SUBDIRS:
        s = all_summaries.get(subdir, {})
        print(f"  {subdir}:")
        print(f"    Files: {s.get('success',0):,} ok, {s.get('errors',0)} errors")
        print(f"    Opcodes: {s.get('total_opcodes',0):,}")
        print(f"    Edges: {s.get('total_edges',0):,}")
        print(f"    Unique param types: {s.get('unique_param_types', [])}")

    # Save global summary JSON
    summary_path = os.path.join(base_dir, "param_features_all_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"\nGlobal summary saved to: {summary_path}")


if __name__ == "__main__":
    import time
    main()
