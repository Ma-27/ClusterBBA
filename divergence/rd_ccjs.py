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
from config import SCALE_DELTA, SCALE_EPSILON
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

def _sigmoid(x: float) -> float:
    """数值稳定的 Sigmoid 函数。"""
    if x >= 50:
        return 1.0
    if x <= -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _scale_weights(cluster: Cluster, delta: float = SCALE_DELTA, epsilon: float = SCALE_EPSILON) -> Dict[
    FrozenSet[str], float]:
    r"""
    对簇中每条 BBA 的 ``m_i(A)`` 使用平滑 Sigmoid 函数，如果远大于 ``delta`` 则接近 1。

    ``h(m) = 1 / (1 + exp(-(m - delta) / epsilon))``

    累加得到 :math:`\\widetilde n(A)`，再在所有焦元上归一化。
    ``delta`` 控制平滑阈值，``epsilon`` 控制过渡的宽度。
    """

    # 收集簇内出现过的全部焦元集合
    focal_sets = set()
    for _, bba in cluster.get_bbas():
        focal_sets.update(bba.keys())

    # 获取 BBA 非空焦元的集合
    votes: Dict[FrozenSet[str], float] = {fs: 0.0 for fs in focal_sets}
    for _, bba in cluster.get_bbas():
        for fs in focal_sets:
            mass = bba.get(fs, 0.0)
            vote = _sigmoid((mass - delta) / epsilon)  # 使用sigmoid替换
            votes[fs] += vote

    total = sum(votes.values())
    if total == 0:
        return {fs: 0.0 for fs in focal_sets}
    return {fs: v / total for fs, v in votes.items()}


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

def rd_ccjs_metric(clus_p: Cluster, clus_q: Cluster, H: int, delta: float = SCALE_DELTA,
                   epsilon: float = SCALE_EPSILON, ) -> float:
    """计算两个簇之间的 ``RD_CCJS`` 距离。

    Parameters
    ----------
    clus_p, clus_q : Cluster
        待测的两个质量函数簇。
    H : int
        全局分形阶。
    delta, epsilon : float
        权重平滑参数，对应 ``config.py`` 中的 ``SCALE_DELTA`` 与 ``SCALE_EPSILON``。
    """

    w_p = _scale_weights(clus_p, delta=delta, epsilon=epsilon)
    w_q = _scale_weights(clus_q, delta=delta, epsilon=epsilon)
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

def divergence_matrix(clusters: List[Cluster], delta: float = SCALE_DELTA,
                      epsilon: float = SCALE_EPSILON, ) -> pd.DataFrame:
    """生成 ``RD_CCJS`` 距离矩阵，可自定义平滑参数。"""
    names = [c.name for c in clusters]
    size = len(clusters)
    H = _max_fractal_order(clusters)
    # 初始化一个距离矩阵
    mat = [[0.0] * size for _ in range(size)]
    # 计算每对簇之间的距离
    for i in range(size):
        for j in range(i + 1, size):
            d = rd_ccjs_metric(clusters[i], clusters[j], H, delta=delta, epsilon=epsilon)
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
