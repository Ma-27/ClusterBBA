# -*- coding: utf-8 -*-
"""
D_CCJS Divergence & Metric Calculator Module
============================================
提供 Cluster‑to‑Cluster Jensen–Shannon 散度 D_CCJS 及其对应度量 (Metric) 的可导入接口。

接口：
- d_ccjs_divergence(m_p, m_q, n_p, n_q) -> float
- d_ccjs_metric(m_p, m_q, n_p, n_q) -> float
- divergence_matrix(bbas, sizes) -> pd.DataFrame
- metric_matrix(bbas, sizes) -> pd.DataFrame
- save_csv(dist_df, out_path=None, default_name='Example_0_1.csv', label='divergence') -> None
- plot_heatmap(dist_df, out_path=None, default_name='Example_0_1.csv', title=None, label='divergence') -> None

脚本用法：
```bash
$ python d_ccjs.py [Example_0_1.csv]
```
示例：
```python
from d_ccjs import load_bbas, divergence_matrix, metric_matrix, save_csv, plot_heatmap
import pandas as pd

df = pd.read_csv('data/examples/Example_0_1.csv')
bbas, _ = load_bbas(df)
# 手动提供簇规模
sizes = {'Clus_1': 10, 'Clus_2': 7}
div_df = divergence_matrix(bbas, sizes)
met_df = metric_matrix(bbas, sizes)
print(div_df, met_df)
save_csv(div_df, default_name='Example_0_1.csv', label='divergence')
save_csv(met_df, default_name='Example_0_1.csv', label='metric')
plot_heatmap(div_df, default_name='Example_0_1.csv', title='D_CCJS Divergence Heatmap', label='divergence')
plot_heatmap(met_df, default_name='Example_0_1.csv', title='D_CCJS Metric Heatmap', label='metric')
```
"""

import math
import os
from typing import Dict, FrozenSet, List, Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd

from config import EPS
# 依赖本项目内现成工具函数 / 模块
from divergence.metric_test import test_nonnegativity, test_symmetry, test_triangle_inequality  # type: ignore


# 计算两簇之间的 D_CCJS divergence（新定义）
def d_ccjs_divergence(
        m_p: Dict[FrozenSet[str], float],
        m_q: Dict[FrozenSet[str], float],
        n_p: int,
        n_q: int
) -> float:
    # 规模权重 w_p, w_q
    w_p = n_p / (n_p + n_q)
    w_q = n_q / (n_p + n_q)

    # 遍历所有焦元
    keys = set(m_p) | set(m_q)
    div = 0.0
    for A in keys:
        # fracional average BBA values
        p = m_p.get(A, 0.0) or EPS
        q = m_q.get(A, 0.0) or EPS
        # 混合分布 M(A) = 0.5 * p + 0.5 * q
        M = 0.5 * p + 0.5 * q
        # 按新公式累加
        div += w_p * p * math.log(p / M, 2) + w_q * q * math.log(q / M, 2)
    # 散度范围裁剪到 [0,1]
    return max(0.0, min(div, 1.0))


# fixme 对应度量 = 根号(divergence),考虑其他构造度量的方式
# 目前的构造度量 (Metric)：参考 RB 论文，将 CCJS 散度转换为度量
# RB_XY = sqrt(|D(XX) + D(YY) - 2 D(XY)| / 2)
def d_ccjs_metric(
        m_p: Dict[FrozenSet[str], float],
        m_q: Dict[FrozenSet[str], float],
        n_p: int,
        n_q: int
) -> float:
    # 第一种构造方式，直接取根号 CCJS
    # return math.sqrt(d_ccjs_divergence(m_p, m_q, n_p, n_q))

    # 第二种方式，参考 RB 论文构造度量
    d_pp = d_ccjs_divergence(m_p, m_p, n_p, n_p)
    d_qq = d_ccjs_divergence(m_q, m_q, n_q, n_q)
    d_pq = d_ccjs_divergence(m_p, m_q, n_p, n_q)
    return math.sqrt(abs(d_pp + d_qq - 2 * d_pq) / 2)


# 生成对称 D_CCJS divergence 矩阵
def divergence_matrix(
        bbas: List[Tuple[str, Dict[FrozenSet[str], float]]],
        sizes: Dict[str, int]
) -> pd.DataFrame:
    names = [name for name, _ in bbas]
    size = len(names)
    mat = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            mat[i][j] = mat[j][i] = d_ccjs_divergence(bbas[i][1], bbas[j][1], sizes[names[i]], sizes[names[j]])
    return pd.DataFrame(mat, index=names, columns=names).round(4)


# 生成 D_CCJS metric 矩阵
def metric_matrix(
        bbas: List[Tuple[str, Dict[FrozenSet[str], float]]],
        sizes: Dict[str, int]
) -> pd.DataFrame:
    names = [name for name, _ in bbas]
    size = len(names)
    mat = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            mat[i][j] = mat[j][i] = d_ccjs_metric(bbas[i][1], bbas[j][1], sizes[names[i]], sizes[names[j]])
    return pd.DataFrame(mat, index=names, columns=names).round(4)


# 保存矩阵为 CSV，可指定 divergence 或 metric
def save_csv(
        dist_df: pd.DataFrame,
        out_path: Optional[str] = None,
        default_name: str = 'Example_0_1.csv',
        label: str = 'divergence',
        index_label: str = 'Cluster'
) -> None:
    if out_path is None:
        # 保存到项目根目录下的 experiments_result
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        res = os.path.join(base, 'experiments_result')
        os.makedirs(res, exist_ok=True)
        fname = f"d_ccjs_{label}_{os.path.splitext(default_name)[0]}.csv"
        out_path = os.path.join(res, fname)
    dist_df.to_csv(out_path, float_format='%.4f', index_label=index_label)


# 绘制并保存热力图
def plot_heatmap(
        dist_df: pd.DataFrame,
        out_path: Optional[str] = None,
        default_name: str = 'Example_0_1.csv',
        title: Optional[str] = None,
        label: str = 'divergence'
) -> None:
    if title is None:
        title = f"D_CCJS {label.title()} Heatmap"
    if out_path is None:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        res = os.path.join(base, 'experiments_result')
        os.makedirs(res, exist_ok=True)
        fname = f"d_ccjs_{label}_{os.path.splitext(default_name)[0]}.png"
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
