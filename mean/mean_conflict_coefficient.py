# -*- coding: utf-8 -*-
"""平均冲突系数计算器
=================

读取包含 BBA 的 CSV 文件，计算其中所有 BBA 之间的 Dempster 冲突系数
``K``，并输出这些系数的算术平均值。

用法::

    $ python mean_conflict_coefficient.py
    $ python mean_conflict_coefficient.py your_file.csv

模块接口：
- ``load_bba_rows(path: str) -> Tuple[List[BBA], List[str]]``
- ``average_conflict(bba_list: List[BBA]) -> float``
- ``compute_avg_conflict_from_csv(csv_path: str) -> float``
"""

import os
import sys
from itertools import combinations
from typing import Dict, FrozenSet, List, Tuple

import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from fusion.ds_rule import conflict_coefficient
from utility.bba import BBA

__all__ = [
    "load_bba_rows",
    "average_conflict",
    "compute_avg_conflict_from_csv",
]


# ------------------------------ 工具函数 ------------------------------ #

# 从 CSV 中加载所有 BBA 行，返回 :class:`BBA` 列表与焦元列顺序

def load_bba_rows(path: str) -> Tuple[List[BBA], List[str]]:
    df = pd.read_csv(path)
    focal_cols = list(df.columns)[1:]
    bba_list: List[BBA] = []
    for _, row in df.iterrows():
        mass: Dict[FrozenSet[str], float] = {}
        for col in focal_cols:
            fs = BBA.parse_focal_set(col)
            mass[fs] = float(row[col])
        bba_list.append(BBA(mass))
    return bba_list, focal_cols


# ------------------------------ 核心函数 ------------------------------ #

# 计算列表中所有两两 BBA 组合的平均冲突系数 ``K``

def average_conflict(bba_list: List[BBA]) -> float:
    n = len(bba_list)
    if n <= 1:
        return 0.0
    ks = [conflict_coefficient(m1, m2) for m1, m2 in combinations(bba_list, 2)]
    return float(sum(ks) / len(ks))


def compute_avg_conflict_from_csv(csv_path: str) -> float:
    bbas, _ = load_bba_rows(csv_path)
    return average_conflict(bbas)


# ------------------------------ 主程序 ------------------------------ #

if __name__ == "__main__":
    # todo 默认配置，根据不同的 CSV 文件或 BBA 簇修改
    default_csv = 'Example_3_2.csv'
    csv_name = sys.argv[1] if len(sys.argv) > 1 else default_csv

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    csv_path = os.path.join(base_dir, 'data', 'examples', csv_name)
    if not os.path.isfile(csv_path):
        print(f'默认 CSV 文件不存在: {default_csv}')
        sys.exit(1)

    avg_k = compute_avg_conflict_from_csv(csv_path)

    print("\n----- 平均 Dempster 冲突系数 K -----")
    print(f"{avg_k:.4f}")
