# -*- coding: utf-8 -*-
r"""fractal_max_entropy.py

实现文档中定义的最大熵分形运算 F，将父焦元的质量按\(2^{|A_i|}-1\) 比例拆分到所有非空子集。
提供与 fractal_average/fractal_hobpa 相同的接口，可计算任意阶分形 BBA。
"""

import itertools
import os
import sys
from typing import Dict, FrozenSet, List, Tuple, Iterator

import pandas as pd

from config import SEG_DEPTH
from utility.bba import BBA
from utility.io import load_bbas, save_bbas

__all__ = [
    'powerset', 'split_once', 'higher_order_bba', 'compute_fractal_df'
]


# ------------------------------ 工具函数 ------------------------------ #

# 生成 s 的所有 *非空* 子集（包括自身）。
def powerset(s: FrozenSet[str]) -> Iterator[FrozenSet[str]]:
    items = list(s)
    for r in range(1, len(items) + 1):
        for combo in itertools.combinations(items, r):
            yield frozenset(combo)


# ------------------------------ 分形核心 ------------------------------ #

# 执行一次最大熵分形拆分。
def split_once(bba: BBA, _h: int) -> BBA:
    new_mass: Dict[FrozenSet[str], float] = {}
    for Ak, mass in bba.items():
        if len(Ak) == 0:
            # 空集质量始终为 0，跳过
            continue
        subsets = bba.subsets_of(Ak, include_empty=False)
        denom = sum(BBA.subset_cardinality(B) for B in subsets)
        for Ai in subsets:
            factor = BBA.subset_cardinality(Ai) / denom
            new_mass[Ai] = new_mass.get(Ai, 0.0) + mass * factor
    return BBA(new_mass, frame=bba.frame)


# h 阶分形，通过迭代 split_once 实现。
def higher_order_bba(bba: BBA, h: int) -> BBA:
    if h == 0:
        return bba
    current = bba
    for _ in range(h):
        current = split_once(current, h)
    return current


# ------------------------------ I/O 与展示 ------------------------------ #

# 计算分形 BBA 的 DataFrame，包含名称和各焦元质量
def compute_fractal_df(
        bbas: List[Tuple[str, BBA]],
        focal_cols: List[str],
        h: int
) -> pd.DataFrame:
    rows = []
    for name, bba in bbas:
        fbba = higher_order_bba(bba, h)
        rows.append([f"{name}_h{h}"] + fbba.to_series(focal_cols))
    return pd.DataFrame(rows, columns=["BBA"] + focal_cols).round(4)


# ------------------------------ 主函数 ------------------------------ #
if __name__ == '__main__':
    # todo 默认示例文件名，可根据实际情况修改
    default_name = 'Example_0.csv'
    csv_name = sys.argv[1] if len(sys.argv) > 1 else default_name
    try:
        h = int(sys.argv[2]) if len(sys.argv) > 2 else SEG_DEPTH
        if h < 0:
            raise ValueError
    except ValueError:
        print('参数 h 必须是不小于 0 的整数。')
        sys.exit(1)

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, '..', 'data', 'examples', csv_name)
    if not os.path.isfile(csv_path):
        print(f'找不到 CSV 文件: {csv_path}')
        sys.exit(1)

    df = pd.read_csv(csv_path)
    bbas, _ = load_bbas(df)
    if bbas:
        theta_sets = bbas[0][1].theta_powerset()
        focal_cols = [BBA.format_set(fs) for fs in theta_sets]
    else:
        focal_cols = []

    out_df = None
    for k in range(0, h + 1):
        print(f'----- 分形结果 (h = {k}) -----')
        out_df = compute_fractal_df(bbas, focal_cols, k)
        print(out_df.to_string(index=False))

    result_dir = os.path.join(base, '..', 'experiments_result')
    os.makedirs(result_dir, exist_ok=True)
    result_file = f'fractal_max_entropy_{os.path.splitext(csv_name)[0]}_h{h}.csv'
    result_path = os.path.join(result_dir, result_file)
    final_bbas = [(f"{name}_h{h}", higher_order_bba(bba, h)) for name, bba in bbas]
    save_bbas(final_bbas, focal_cols, out_path=result_path,
              default_name=os.path.basename(result_path))
    print(f'结果 CSV: {result_path}')
