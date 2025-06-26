# -*- coding: utf-8 -*-
"""
BJS Divergence Calculator Module
================================
提供 Belief Jensen–Shannon divergence (BJS) 的可导入接口，供其他脚本调用。

接口：
- bjs_divergence(m1, m2) -> float
- bjs_metric(m1, m2) -> float
- divergence_matrix(bbas) -> pd.DataFrame
- metric_matrix(bbas) -> pd.DataFrame
- save_csv(dist_df, out_path=None, default_name, label) -> None
- plot_heatmap(dist_df, out_path=None, default_name, title) -> None

示例：
```python
from bjs_distance import load_bbas, divergence_matrix, metric_matrix, save_csv, plot_heatmap
import pandas as pd

df = pd.read_csv('data/examples/Example_3_3.csv')
bbas, _ = load_bbas(df)
div_df = divergence_matrix(bbas)
met_df = metric_matrix(bbas)
save_csv(div_df, default_name='Example_3_3.csv', label='divergence')
save_csv(met_df, default_name='Example_3_3.csv', label='metric')
plot_heatmap(div_df, default_name='Example_3_3.csv', title='BJS Divergence Heatmap')
plot_heatmap(met_df, default_name='Example_3_3.csv', title='BJS Metric Heatmap')
```
"""

import math
import os
from typing import Dict, FrozenSet, List, Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from divergence.metric_test import test_nonnegativity, test_symmetry, test_triangle_inequality  # type: ignore

# ------------------------------ 常量 ------------------------------ #
EPS = 1e-12  # 避免 log(0)


# 计算两条 BBA 之间的 BJS divergence
def bjs_divergence(
        m1: Dict[FrozenSet[str], float],
        m2: Dict[FrozenSet[str], float]
) -> float:
    keys = set(m1) | set(m2)
    div = 0.0
    for A in keys:
        p = m1.get(A, 0.0) or EPS
        q = m2.get(A, 0.0) or EPS
        m = 0.5 * (p + q)
        div += 0.5 * p * math.log(p / m, 2) + 0.5 * q * math.log(q / m, 2)
    # BJS 的取值范围应在 [0, 1]
    return max(0.0, min(div, 1.0))


# 对应度量 = 根号(divergence)
def bjs_metric(
        m1: Dict[FrozenSet[str], float],
        m2: Dict[FrozenSet[str], float]
) -> float:
    return math.sqrt(bjs_divergence(m1, m2))


# 生成对称 BJS 距离矩阵 DataFrame
def divergence_matrix(bbas: List[Tuple[str, Dict[FrozenSet[str], float]]]) -> pd.DataFrame:
    names = [n for n, _ in bbas]
    size = len(names)
    mat = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            d = bjs_divergence(bbas[i][1], bbas[j][1])
            mat[i][j] = mat[j][i] = d
    return pd.DataFrame(mat, index=names, columns=names).round(4)


# 生成 metric 矩阵
def metric_matrix(bbas: List[Tuple[str, Dict[FrozenSet[str], float]]]) -> pd.DataFrame:
    names = [n for n, _ in bbas]
    size = len(names)
    mat = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            m = bjs_metric(bbas[i][1], bbas[j][1])
            mat[i][j] = mat[j][i] = m
    return pd.DataFrame(mat, index=names, columns=names).round(4)


# 保存距离矩阵为 CSV 文件，可指定输出路径或使用默认文件名
def save_csv(
        dist_df: pd.DataFrame,
        out_path: Optional[str] = None,
        default_name: str = 'Example_3_3.csv',
        label: str = 'divergence',
        index_label: str = 'BBA'
) -> None:
    if out_path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        res = os.path.join(base, '..', 'experiments_result')
        os.makedirs(res, exist_ok=True)
        fname = f"bjs_{label}_{os.path.splitext(default_name)[0]}.csv"
        out_path = os.path.join(res, fname)
    dist_df.to_csv(out_path, float_format='%.4f', index_label=index_label)


# 绘制并保存热力图
def plot_heatmap(
        dist_df: pd.DataFrame,
        out_path: Optional[str] = None,
        default_name: str = 'Example_3_3.csv',
        title: Optional[str] = 'BJS Divergence Heatmap',
        label: str = 'divergence'
) -> None:
    if out_path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        res = os.path.join(base, '..', 'experiments_result')
        os.makedirs(res, exist_ok=True)
        fname = f"bjs_{label}_{os.path.splitext(default_name)[0]}.png"
        out_path = os.path.join(res, fname)
    fig, ax = plt.subplots()
    cax = ax.matshow(dist_df.values)
    fig.colorbar(cax)
    ax.set_xticks(range(len(dist_df)))
    ax.set_xticklabels(dist_df.columns, rotation=90)
    ax.set_yticks(range(len(dist_df)))
    ax.set_yticklabels(dist_df.index)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
