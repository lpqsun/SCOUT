#!/usr/bin/env python3
"""
智能合约漏洞轨迹数据清洗脚本

清洗规则:
1. 删除缺失签名操作码的 trace（如 timestamp 类别中无 TIMESTAMP opcode）
2. 跨类别去重：相同 opcode 序列出现在多个类别时，按实际特征保留最合适的类别
3. 长度过滤：删除 < 20 步或 > 3000 步的痕迹

执行: python3 clean_traces.py          # dry-run, 仅报告
      python3 clean_traces.py --apply  # 实际执行删除
"""

import json, glob, os, shutil, sys, argparse
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

CATEGORIES = {
    'reentrancy':                          f'{BASE}/reentrancy_ParamFeatures',
    'timestamp_dependency':                f'{BASE}/timestamp_dependency_ParamFeatures',
    'integer_overflow':                   f'{BASE}/integer_overflow_ParamFeatures',
    'unchecked_external_call':            f'{BASE}/unchecked_external_call_ParamFeatures',
    'balance_check':                      f'{BASE}/balance_check_ParamFeatures',
    'block_number_dependency':            f'{BASE}/block_number_dependency_ParamFeatures',
}

SIGNATURE_OPCODES = {
    'reentrancy':                 ['CALL'],
    'timestamp_dependency':       ['TIMESTAMP'],
    'integer_overflow':           ['ADD', 'MUL', 'SUB', 'EXP', 'ADDMOD', 'MULMOD', 'SMOD'],
    'unchecked_external_call':    ['CALL', 'DELEGATECALL', 'CALLCODE'],
    'balance_check':              ['BALANCE', 'SELFBALANCE'],
    'block_number_dependency':    ['NUMBER'],
}

LABEL_SHORT = {
    'reentrancy': 'RE',
    'timestamp_dependency': 'TS',
    'integer_overflow': 'IO',
    'unchecked_external_call': 'UC',
    'balance_check': 'BAL',
    'block_number_dependency': 'BN',
}

# ---------- helpers ----------
def load_trace(json_path):
    with open(json_path) as f:
        data = json.load(f)
    fn = os.path.basename(json_path)
    ops = [s['opcode'] for s in data['steps']]
    op_set = set(ops)
    op_hash = hash(''.join(ops))
    return {
        'file': json_path,
        'dirname': os.path.basename(os.path.dirname(json_path)),
        'basename': fn,
        'steps': data['total_steps'],
        'ops': ops,
        'op_set': op_set,
        'op_hash': op_hash,
        'metadata': data.get('metadata', {}),
        'data': data,
    }

def has_signature(trace, cat):
    sig = SIGNATURE_OPCODES[cat]
    return any(op in trace['op_set'] for op in sig)

def get_actual_categories(op_set):
    """根据实际 opcodes 判断 trace 可能属于哪些类别"""
    cats = []
    if 'BALANCE' in op_set or 'SELFBALANCE' in op_set:
        cats.append('balance_check')
    if 'CALL' in op_set:
        if 'SSTORE' in op_set:
            cats.append('reentrancy')
        else:
            cats.append('unchecked_external_call')
    if 'NUMBER' in op_set:
        cats.append('block_number_dependency')
    if 'TIMESTAMP' in op_set:
        cats.append('timestamp_dependency')
    if any(op in op_set for op in ['ADD', 'MUL', 'SUB', 'EXP']):
        cats.append('integer_overflow')
    if 'DELEGATECALL' in op_set or 'CALLCODE' in op_set:
        if 'unchecked_external_call' not in cats:
            cats.append('unchecked_external_call')
    return cats

def resolve_duplicates(traces_by_hash, cat_trace_map):
    """
    对于相同的 op_hash，决定应该保留在哪个类别中。
    返回: {trace_hash: set(categories_to_keep)}
    """
    resolution = {}

    for op_hash, cat_traces in traces_by_hash.items():
        if len(cat_traces) <= 1:
            continue

        present_cats = set(cat_traces.keys())
        # 获取实际的 opcode 特征（所有同名 trace 的 opcode 一样）
        sample_trace = list(cat_traces.values())[0]
        op_set = sample_trace['op_set']

        # 确定 trace 实际支持哪些类别
        actual_cats = set(get_actual_categories(op_set))

        # 规则：只保留在 actual_cats 中存在的类别
        # 优先级: BAL > RE > BN > TS > UC > IO
        priority = ['balance_check', 'reentrancy', 'block_number_dependency',
                    'timestamp_dependency', 'unchecked_external_call', 'integer_overflow']

        keep = None
        for cat in priority:
            if cat in present_cats and cat in actual_cats:
                keep = cat
                break

        # 如果所有优先级都不匹配（理论上不应出现），保留第一个有签名的
        if keep is None:
            for cat in priority:
                if cat in present_cats and has_signature(sample_trace, cat):
                    keep = cat
                    break

        # 如果还是没有（极罕见：trace 在多个类别但没有匹配任何签名），全部删除
        resolution[op_hash] = keep  # None = delete from all

    return resolution

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true', help='实际执行删除/移动操作')
    args = parser.parse_args()

    print("=" * 70)
    print("智能合约漏洞轨迹数据清洗")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模式: {'🔴 实际执行' if args.apply else '🟡 DRY-RUN (仅报告)'}")
    print("=" * 70)

    # 1. 加载所有 trace
    print("\n[1/5] 加载所有轨迹...")
    all_traces = defaultdict(list)  # cat -> [trace dicts]
    for cat, dirpath in CATEGORIES.items():
        files = sorted(glob.glob(os.path.join(dirpath, '*_param_features.json')))
        for f in files:
            t = load_trace(f)
            all_traces[cat].append(t)
        print(f"  {LABEL_SHORT[cat]:>4s} ({cat}): {len(all_traces[cat])} traces")

    total_before = sum(len(v) for v in all_traces.values())
    print(f"  总计: {total_before}")

    # 2. 签名 opcode 缺失检查
    print("\n[2/5] 检查签名操作码缺失...")
    no_sig = defaultdict(list)
    for cat, traces in all_traces.items():
        for t in traces:
            if not has_signature(t, cat):
                no_sig[cat].append(t)
        pct = 100 * len(no_sig[cat]) / len(traces) if traces else 0
        print(f"  {LABEL_SHORT[cat]:>4s}: {len(no_sig[cat])}/{len(traces)} 缺失签名 ({pct:.1f}%)")

    total_no_sig = sum(len(v) for v in no_sig.values())
    print(f"  总计缺失签名: {total_no_sig}")

    # 3. 跨类别完全相同的 trace
    print("\n[3/5] 检查跨类别重复trace...")

    # 构建 op_hash -> {cat: trace}
    hash_map = defaultdict(lambda: {})
    for cat, traces in all_traces.items():
        for t in traces:
            hash_map[t['op_hash']][cat] = t

    # 只关注出现在多个类别的 hash
    multi_cat_hashes = {h: cats for h, cats in hash_map.items() if len(cats) > 1}
    print(f"  出现在多个类别的 trace: {len(multi_cat_hashes)}")

    # 按类别对统计
    dup_stats = Counter()
    for h, cats in multi_cat_hashes.items():
        cat_list = sorted(cats.keys())
        for i, c1 in enumerate(cat_list):
            for c2 in cat_list[i+1:]:
                dup_stats[(c1, c2)] += 1

    for (c1, c2), cnt in dup_stats.most_common():
        print(f"    {LABEL_SHORT[c1]:>4s} ∩ {LABEL_SHORT[c2]:<4s}: {cnt}")

    # 4. 解析重复
    print("\n[4/5] 解析去重规则...")
    resolution = resolve_duplicates(hash_map, all_traces)

    dup_remove_count = Counter()
    dup_keep_count = Counter()
    for h, keep_cat in resolution.items():
        if keep_cat is None:
            # 应该从所有类别中删除
            for cat in hash_map[h]:
                dup_remove_count[cat] += 1
        else:
            for cat in hash_map[h]:
                if cat != keep_cat:
                    dup_remove_count[cat] += 1
                else:
                    dup_keep_count[cat] += 1

    print("  各分类因重复删除数量:")
    for cat in CATEGORIES:
        print(f"    {LABEL_SHORT[cat]:>4s}: 删除 {dup_remove_count.get(cat, 0)}, 保留 {dup_keep_count.get(cat, 0)}")

    # 5. 长度过滤
    print("\n[5/5] 长度过滤 (< 20 or > 3000)...")
    too_short = defaultdict(list)
    too_long = defaultdict(list)
    for cat, traces in all_traces.items():
        for t in traces:
            if t['steps'] < 20:
                too_short[cat].append(t)
            elif t['steps'] > 3000:
                too_long[cat].append(t)

    total_short = sum(len(v) for v in too_short.values())
    total_long = sum(len(v) for v in too_long.values())
    for cat in CATEGORIES:
        s, l = len(too_short.get(cat, [])), len(too_long.get(cat, []))
        if s + l > 0:
            print(f"  {LABEL_SHORT[cat]:>4s}: <20={s}, >3000={l}")
    print(f"  总计: <20={total_short}, >3000={total_long}")

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    print("清洗汇总")
    print("=" * 70)

    # 每个类别综合决策: {cat: set(trace_key_to_remove)}
    to_remove = defaultdict(set)

    # 规则1: 无签名 opcode
    for cat, traces in no_sig.items():
        for t in traces:
            to_remove[cat].add(t['basename'])

    # 规则2: 重复
    for h, keep_cat in resolution.items():
        if keep_cat is None:
            for cat in hash_map[h]:
                to_remove[cat].add(hash_map[h][cat]['basename'])
        else:
            for cat in hash_map[h]:
                if cat != keep_cat:
                    to_remove[cat].add(hash_map[h][cat]['basename'])

    # 规则3: 长度过滤
    for cat, traces in too_short.items():
        for t in traces:
            to_remove[cat].add(t['basename'])
    for cat, traces in too_long.items():
        for t in traces:
            to_remove[cat].add(t['basename'])

    # 打印每个类别
    print(f"\n{'类别':<30s} {'原始':>6s} {'删除':>6s} {'保留':>6s} {'保留率':>7s}")
    print("-" * 60)
    grand_remove = 0
    for cat in CATEGORIES:
        original = len(all_traces[cat])
        removed = len(to_remove[cat])
        kept = original - removed
        rate = 100 * kept / original if original > 0 else 0
        grand_remove += removed
        print(f"{cat:<30s} {original:>6d} {removed:>6d} {kept:>6d} {rate:>6.1f}%")

    print("-" * 60)
    print(f"{'总计':<30s} {total_before:>6d} {grand_remove:>6d} {total_before-grand_remove:>6d} {100*(total_before-grand_remove)/total_before:>6.1f}%")

    # 删除原因统计
    print(f"\n删除原因统计:")
    print(f"  无签名opcode: {total_no_sig}")
    dup_total = sum(dup_remove_count.values())
    print(f"  跨类重复: {dup_total}")
    print(f"  长度 < 20: {total_short}")
    print(f"  长度 > 3000: {total_long}")
    print(f"  (注: 一个trace可能满足多个条件，去重后总删除 < 以上之和)")

    # ---- 执行 ----
    if not args.apply:
        print(f"\n🟡 DRY-RUN 完成。使用 --apply 实际执行删除。")
        return

    print(f"\n🔴 实际执行删除...")

    # 创建备份目录
    backup_dir = os.path.join(BASE, '_cleaned_backup')
    os.makedirs(backup_dir, exist_ok=True)

    removed_count = 0
    for cat in CATEGORIES:
        dirpath = CATEGORIES[cat]
        cat_removed = 0
        cat_backup = os.path.join(backup_dir, os.path.basename(dirpath))
        os.makedirs(cat_backup, exist_ok=True)

        for bn in to_remove[cat]:
            # 每个 trace 有 5 个文件
            base = bn.replace('_param_features.json', '')
            for suffix in ['_param_features.json', '_param_features.txt',
                          '_param_only.txt', '_dep_matrix.json', '_stats.npy']:
                fname = base + suffix
                src = os.path.join(dirpath, fname)
                if os.path.exists(src):
                    dst = os.path.join(cat_backup, fname)
                    shutil.move(src, dst)
                    cat_removed += 1

        # 清理备份目录中空的 trace（只有部分文件）
        cat_removed_traces = len(to_remove[cat])
        removed_count += cat_removed_traces
        print(f"  {LABEL_SHORT[cat]:>4s}: 删除 {cat_removed_traces} traces ({cat_removed} files) → {cat_backup}")

    print(f"\n✅ 清洗完成！删除 {removed_count} traces，备份至 {backup_dir}")
    print(f"   剩余: {total_before - removed_count} traces")

    # 生成报告
    report_path = os.path.join(BASE, '_cleaned_backup', 'cleanup_report.json')
    report = {
        'timestamp': datetime.now().isoformat(),
        'before': {cat: len(all_traces[cat]) for cat in CATEGORIES},
        'removed': {cat: len(to_remove[cat]) for cat in CATEGORIES},
        'after': {cat: len(all_traces[cat]) - len(to_remove[cat]) for cat in CATEGORIES},
        'reasons': {
            'no_signature': {cat: len(no_sig[cat]) for cat in CATEGORIES},
            'duplicate': {cat: dup_remove_count.get(cat, 0) for cat in CATEGORIES},
            'too_short': {cat: len(too_short.get(cat, [])) for cat in CATEGORIES},
            'too_long': {cat: len(too_long.get(cat, [])) for cat in CATEGORIES},
        }
    }
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"   报告: {report_path}")

if __name__ == '__main__':
    main()
