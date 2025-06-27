# -*- coding: utf-8 -*-
"""
Average BBA Calculator
==============================

提供计算指定 CSV 文件中所有 BBA 的平均质量值的接口，供其他脚本调用。

对齐焦元后计算该文件中所有的质量（Mass）平均值，并在控制台输出结果。
可通过修改变量来指定需要平均的 CSV 文件名。

在 experiments_result 目录中，针对指定的单个 BBA 文件，

注意：只可用作计算 BBA 平均值！

接口：
- load_bba_rows(path: str) -> Tuple[List[Dict[FrozenSet[str], float]], List[str]]
- compute_avg_bba(bba_list: List[Dict[FrozenSet[str], float]]]) -> Dict[FrozenSet[str], float]
- bba_to_series(bba: Dict[FrozenSet[str], float], order: List[str]) -> List[float]
- prepare_avg_dataframe(avg_bba: Dict[FrozenSet[str], float], focal_order: List[str]) -> pd.DataFrame

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
from typing import Dict, FrozenSet, List, Tuple

import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from utility.io import parse_focal_set, format_set

__all__ = ['load_bba_rows', 'compute_avg_bba', 'bba_to_series', 'prepare_avg_dataframe']


# ------------------------------ 工具函数 ------------------------------ #

# 从 CSV 中加载所有 BBA 行，返回 BBA 字典列表与焦元列顺序"
def load_bba_rows(path: str) -> Tuple[List[Dict[FrozenSet[str], float]], List[str]]:
    df = pd.read_csv(path)
    # 第一列为标签，后续列为焦元质量值
    focal_cols = list(df.columns)[1:]
    bba_list: List[Dict[FrozenSet[str], float]] = []
    # 遍历每一行，提取质量值
    for _, row in df.iterrows():
        bba: Dict[FrozenSet[str], float] = {}
        for col in focal_cols:
            fs = parse_focal_set(col)
            bba[fs] = float(row[col])  # 将质量值转换为浮点数
        bba_list.append(bba)
    return bba_list, focal_cols


# 计算 BBA 列表中每个焦元的平均质量"
def compute_avg_bba(bba_list: List[Dict[FrozenSet[str], float]]) -> Dict[FrozenSet[str], float]:
    if not bba_list:
        return {}
    mass_sum: Dict[FrozenSet[str], float] = {}
    for bba in bba_list:
        for fs, m in bba.items():
            mass_sum[fs] = mass_sum.get(fs, 0.0) + m
    count = len(bba_list)
    return {fs: mass / count for fs, mass in mass_sum.items()}


def bba_to_series(bba: Dict[FrozenSet[str], float], order: List[str]) -> List[float]:
    """按给定的原始列名顺序生成质量值列表，用于输出和 DataFrame 构造"""
    data = {format_set(fs): mass for fs, mass in bba.items()}
    # 对齐顺序，不存在则填 0.0
    return [data.get(col, 0.0) for col in order]


def prepare_avg_dataframe(avg_bba: Dict[FrozenSet[str], float], focal_order: List[str]) -> pd.DataFrame:
    """构造包含平均质量的 DataFrame，并保留4位小数"""
    header = ['Label'] + focal_order
    row = ['avg'] + bba_to_series(avg_bba, focal_order)
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
