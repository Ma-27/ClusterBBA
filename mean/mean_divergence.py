# -*- coding: utf-8 -*-
"""
Average BJS Divergence Calculator
=================================
读取由 bjs.py 生成的对称散度矩阵 (CSV)，计算并输出
所有非对角元（上三角即可）的算术平均值。

用法（在 experiments 目录下执行）：
$ python mean_divergence.py                       # 默认 bjs_Example_3_3.csv
$ python mean_divergence.py bjs_Example_2_3.csv   # 指定文件

模块接口：
- average_divergence(dist_df: pd.DataFrame) -> float
- compute_avg_divergence_from_csv(csv_path: str) -> float
"""

import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd

__all__ = ['average_divergence', 'compute_avg_divergence_from_csv']


# ------------------------------ 工具函数 ------------------------------ #

def load_divergence_matrix(csv_path: str) -> pd.DataFrame:
    """从 CSV 文件加载对称散度矩阵，第一列为索引"""
    return pd.read_csv(csv_path, index_col=0)


def compute_avg_divergence_from_csv(csv_path: str) -> float:
    """加载 CSV 并计算平均散度"""
    df = load_divergence_matrix(csv_path)
    return average_divergence(df)


# ------------------------------ 核心函数 ------------------------------ #

# 计算对称散度矩阵 dist_df 的平均散度（排除对角线，自然地等价于只取上三角）。
def average_divergence(dist_df: pd.DataFrame) -> float:
    n: int = dist_df.shape[0]
    if n <= 1:  # 只有一行默认返回 0
        return 0.0
    tri_idx: Tuple[np.ndarray, np.ndarray] = np.triu_indices(n, k=1)
    values = dist_df.values[tri_idx]
    return float(values.mean())


# ------------------------------ 主程序 ------------------------------ #
if __name__ == '__main__':
    # 默认文件名，可通过命令行覆盖，todo 这里可以修改
    default_csv = 'bjs_divergence_Example_0_4.csv'
    csv_name = sys.argv[1] if len(sys.argv) > 1 else default_csv

    # 构造绝对路径：experiments_result/<csv_name>
    base_dir = os.path.dirname(os.path.abspath(__file__))  # …/experiments
    csv_path = os.path.normpath(os.path.join(
        base_dir, '..', 'experiments_result', csv_name))

    if not os.path.isfile(csv_path):
        print(f'找不到散度矩阵 CSV 文件: {csv_path}')
        sys.exit(1)

    # 载入散度矩阵（index 第一列是 BBA 名称）
    dist_df = pd.read_csv(csv_path, index_col=0)

    # 计算平均散度
    avg_div = average_divergence(dist_df)

    # --------- 控制台输出 --------- #
    print('\n----- 平均 BJS 散度 -----')
    print(f'{avg_div:.4f}')
