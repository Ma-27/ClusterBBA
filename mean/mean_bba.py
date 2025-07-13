# -*- coding: utf-8 -*-
"""BBA 平均计算器
=================

提供计算指定 CSV 文件中所有 BBA 的平均质量值的接口。

对齐焦元后取各质量值的平均数并打印结果，可修改变量以指定目标 CSV
文件名。仅用于计算 BBA 的简单平均值。

接口：
- ``load_bba_rows(path: str) -> Tuple[List[BBA], List[str]]``
- ``compute_avg_bba(bba_list: List[BBA]) -> BBA``
- ``prepare_avg_dataframe(avg_bba: BBA, focal_order: List[str]) -> pd.DataFrame``

示例：
```python
from mean_bba import load_bba_rows, compute_avg_bba, prepare_avg_dataframe

bba_list, focal_order = load_bba_rows('path/to/file.csv')
avg_bba = compute_avg_bba(bba_list)
df_avg = prepare_avg_dataframe(avg_bba, focal_order)
print(df_avg)
```
"""

import os
import sys
from functools import reduce
from typing import Dict, FrozenSet, List, Tuple

import numpy as np
import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from utility.bba import BBA

__all__ = ['load_bba_rows', 'compute_avg_bba', 'prepare_avg_dataframe']


# ------------------------------ 工具函数 ------------------------------ #

# 从 CSV 中加载所有 BBA 行，返回 :class:`BBA` 列表与焦元列顺序"
def load_bba_rows(path: str) -> Tuple[List[BBA], List[str]]:
    df = pd.read_csv(path)
    # 第一列为标签，后续列为焦元质量值
    focal_cols = list(df.columns)[1:]
    bba_list: List[BBA] = []
    # 遍历每一行，提取质量值
    for _, row in df.iterrows():
        mass: Dict[FrozenSet[str], float] = {}
        for col in focal_cols:
            fs = BBA.parse_focal_set(col)
            mass[fs] = float(row[col])  # 将质量值转换为浮点数
        bba_list.append(BBA(mass))
    return bba_list, focal_cols


# 计算 BBA 列表中每个焦元的平均质量"
def compute_avg_bba(bba_list: List[BBA]) -> BBA:
    """利用 ``numpy`` 计算 BBA 的逐焦元平均值"""
    if not bba_list:
        return BBA()

    frame = reduce(BBA.union, (b.frame for b in bba_list), frozenset())
    proto = BBA({}, frame=frame)
    focals = list(proto.keys())
    mass_mat = np.array([[b.get_mass(fs) for fs in focals] for b in bba_list])
    avg_values = mass_mat.mean(axis=0)
    avg_mass = {fs: float(v) for fs, v in zip(focals, avg_values)}
    return BBA(avg_mass, frame=frame)


def prepare_avg_dataframe(avg_bba: BBA, focal_order: List[str]) -> pd.DataFrame:
    """构造包含平均质量的 DataFrame，并保留4位小数"""
    header = ['Label'] + focal_order
    row = ['avg'] + avg_bba.to_series(focal_order)
    df_avg = pd.DataFrame([row], columns=header).round(4)
    return df_avg


# ------------------------------ 主函数 ------------------------------ #
if __name__ == '__main__':
    # todo 替换为所需文件名，指定需要计算平均的 CSV 文件名
    target_file = 'fractal_Example_3_3_3_h2.csv'

    # 构造文件路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.normpath(os.path.join(base_dir, '..', 'experiments_result'))
    path = os.path.join(result_dir, target_file)
    if not os.path.isfile(path):
        print(f"未找到文件: {path}")
        sys.exit(1)

    # 加载 CSV 中的所有 BBA 行
    bba_list, focal_order = load_bba_rows(path)
    if not bba_list:
        print("CSV 中未包含任何 BBA 行。")
        sys.exit(1)

    avg_bba = compute_avg_bba(bba_list)
    df_avg = prepare_avg_dataframe(avg_bba, focal_order)

    # 打印结果
    print('----- 指定 CSV BBA 平均 -----')
    print(df_avg.to_string(index=False, float_format="%.4f"))
