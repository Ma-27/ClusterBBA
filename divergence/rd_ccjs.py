# -*- coding: utf-8 -*-
"""RD_CCJS 距离计算模块
=========================

按《研究思路构建》中提出的增强簇‑簇 Jensen–Shannon 散度 ``RD_CCJS``
实现多个 :class:`cluster.Cluster` 对象之间的距离计算。与
``divergence.d_ccjs`` 接口相似，但自动在全局最高分形阶上对齐簇心，
并根据焦元出现频次计算规模权重。

主要接口
--------
- ``rd_ccjs_metric(clus_p, clus_q, H) -> float``
- ``divergence_matrix(clusters) -> pd.DataFrame``
- ``metric_matrix(clusters) -> pd.DataFrame``
- ``save_csv(dist_df, out_path=None, default_name='Example_0_1.csv', label='divergence')``
- ``plot_heatmap(dist_df, out_path=None, default_name='Example_0_1.csv', title=None, label='divergence')``

示例
^^^^
::

    from cluster import initialize_empty_cluster
    from divergence.rd_ccjs import metric_matrix

    # 假设 clus1、clus2 已初始化
    dist_df = metric_matrix([clus1, clus2])
    print(dist_df)
"""

from __future__ import annotations

import math
import os
from typing import Dict, FrozenSet, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd

from cluster.one_cluster import Cluster  # type: ignore
# 分形运算可采用不同的分形办法，默认使用 fractal_average
from fractal.fractal_average import higher_order_bba  # type: ignore

__all__ = [
    "rd_ccjs_metric",
    "divergence_matrix",
    "metric_matrix",
    "save_csv",
    "plot_heatmap",
]


# ---------------------------------------------------------------------------
# 工具函數
# ---------------------------------------------------------------------------

def _scale_weights(cluster: Cluster) -> Dict[FrozenSet[str], float]:
    """计算簇中每个焦元的规模权重 ``w(A)``。"""
    # 该字典用于统计每个焦元（fs）在簇中出现的次数
    counts: Dict[FrozenSet[str], int] = {}
    for _, bba in cluster.get_bbas():
        # 提醒：bba 是一个 Dict[FrozenSet[str], float]，映射焦元到其质量
        for fs, mass in bba.items():
            # 只有质量大于0的焦元才计入统计 todo 之后研究有改进空间
            if mass > 0:
                counts[fs] = counts.get(fs, 0) + 1
    # 计算所有焦元出现的总次数
    total = sum(counts.values())
    # 如果总次数为0（簇中没有任何质量>0的焦元），返回所有焦元权重为0
    if total == 0:
        return {fs: 0.0 for fs in counts}
    # 否则，计算每个焦元的规模权重 = 该焦元出现次数 / 总次数
    return {fs: cnt / total for fs, cnt in counts.items()}


def _aligned_centroid(cluster: Cluster, H: int) -> Dict[FrozenSet[str], float]:
    """将簇心对齐到全局最大分形阶 ``H``。"""
    centroid = cluster.get_centroid() or {}
    diff = max(H - cluster.h, 0)
    if diff == 0:
        return centroid
    return higher_order_bba(centroid, diff)


def _max_fractal_order(clusters: Iterable[Cluster]) -> int:
    """获取所有簇的最大分形阶。"""
    # for c in clusters:
    # print(f"Cluster {c.name} fractal order: {c.h}")
    return max((c.h for c in clusters), default=0)


# ---------------------------------------------------------------------------
# 核心距离计算
# ---------------------------------------------------------------------------

def rd_ccjs_metric(clus_p: Cluster, clus_q: Cluster, H: int) -> float:
    """计算两个簇之间的 ``RD_CCJS`` 距离。"""
    w_p = _scale_weights(clus_p)  # shape: {FrozenSet[str]: float}
    w_q = _scale_weights(clus_q)  # shape: {FrozenSet[str]: float}
    # debug 用，记得删掉
    # print(f"Cluster {clus_p.name} weights w_p: {w_p}")
    # print(f"Cluster {clus_q.name} weights w_q: {w_q}")
    m_p = _aligned_centroid(clus_p, H)
    m_q = _aligned_centroid(clus_q, H)

    # 对所有两个簇之间出现过的非0key取并集
    keys = set(m_p) | set(m_q) | set(w_p) | set(w_q)
    dist = 0.0
    for A in keys:
        phi_p = math.sqrt(w_p.get(A, 0.0) * m_p.get(A, 0.0))
        phi_q = math.sqrt(w_q.get(A, 0.0) * m_q.get(A, 0.0))
        dist += (phi_p - phi_q) ** 2
    return math.sqrt(dist)


# ---------------------------------------------------------------------------
# 矩阵计算
# ---------------------------------------------------------------------------

def divergence_matrix(clusters: List[Cluster]) -> pd.DataFrame:
    """生成 ``RD_CCJS`` 距离矩阵。"""
    names = [c.name for c in clusters]
    size = len(clusters)
    H = _max_fractal_order(clusters)
    # 初始化一个距离矩阵
    mat = [[0.0] * size for _ in range(size)]
    # 计算每对簇之间的距离
    for i in range(size):
        for j in range(i + 1, size):
            d = rd_ccjs_metric(clusters[i], clusters[j], H)
            mat[i][j] = mat[j][i] = d
    return pd.DataFrame(mat, index=names, columns=names).round(4)


# ``metric_matrix`` 与 ``divergence_matrix`` 在此等价，为接口兼容保留
metric_matrix = divergence_matrix


# ---------------------------------------------------------------------------
# I/O 輔助
# ---------------------------------------------------------------------------

def save_csv(
        dist_df: pd.DataFrame,
        out_path: str | None = None,
        default_name: str = "Example_0_1.csv",
        label: str = "divergence",
        index_label: str = "Cluster",
) -> None:
    """将距离矩阵保存为 CSV。"""
    if out_path is None:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        res = os.path.join(base, "experiments_result")
        os.makedirs(res, exist_ok=True)
        fname = f"rd_ccjs_{label}_{os.path.splitext(default_name)[0]}.csv"
        out_path = os.path.join(res, fname)
    dist_df.to_csv(out_path, float_format="%.4f", index_label=index_label)


def plot_heatmap(
        dist_df: pd.DataFrame,
        out_path: str | None = None,
        default_name: str = "Example_0_1.csv",
        title: str | None = None,
        label: str = "divergence",
) -> None:
    """绘制并保存距离矩阵的热力图。"""
    if title is None:
        title = f"RD_CCJS {label.title()} Heatmap"
    if out_path is None:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        res = os.path.join(base, "experiments_result")
        os.makedirs(res, exist_ok=True)
        fname = f"rd_ccjs_{label}_{os.path.splitext(default_name)[0]}.png"
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
