# -*- coding: utf-8 -*-
"""簇集合可视化工具
====================
提供 ``visualize_clusters`` 函数，用于将多簇结构以二维图形方式展示。
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from cluster.one_cluster import Cluster
from divergence.rd_ccjs import divergence_matrix
# 依赖本项目内现成工具函数 / 模块
from utility.plot_style import apply_style
from utility.plot_utils import savefig

apply_style()

__all__ = ["visualize_clusters"]


def _cluster_positions(clusters: List[Cluster], dists: List[List[float]]) -> List[tuple[float, float]]:
    """根据簇间距离估计各簇的位置。"""
    K = len(clusters)
    if K == 0:
        return []
    # 每个簇对应一个角度，按顺时针均分 2π
    angles = [2 * math.pi * i / K for i in range(K)]
    # 计算平均距离，作为半径的线性尺度
    avg_dists = [sum(row) / (K - 1) if K > 1 else 1.0 for row in dists]
    max_dist = max(avg_dists) if avg_dists else 1.0
    radii = [1.0 + 3.0 * (d / max_dist if max_dist > 0 else 0.0) for d in avg_dists]
    # 转换为笛卡尔坐标
    return [(r * math.cos(a), r * math.sin(a)) for r, a in zip(radii, angles)]


def visualize_clusters(
        clusters: Iterable[Cluster],
        save_path: Optional[str] = None,
        show: bool = True,
) -> None:
    """绘制簇集合的关系图。

    Parameters
    ----------
    clusters : Iterable[Cluster]
        需要展示的簇对象列表。
    save_path : str | None
        若指定，则将图像保存到该路径。
    show : bool
        是否直接展示图像窗口。
    """

    clus_list = list(clusters)
    if not clus_list:
        return

    # 计算 RD_CCJS 距离矩阵
    dist_df = divergence_matrix(clus_list)
    dist_mat = dist_df.values.tolist()

    # 计算每个簇的中心位置
    positions = _cluster_positions(clus_list, dist_mat)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal', adjustable='datalim')
    ax.axis('off')

    # 绘制簇与标签
    for (x, y), clus in zip(positions, clus_list):
        n_i = len(clus.get_bbas())
        size = 0.3 + 0.2 * n_i
        cloud = Ellipse((x, y), width=size, height=size * 0.6,
                        edgecolor='black', facecolor='#cfe2f3', alpha=0.6)
        ax.add_patch(cloud)
        # 簇名称
        ax.text(x, y + size * 0.35, clus.name, ha='center', va='bottom', fontsize=10)
        # 元素名称列表
        elems = "\n".join(name for name, _ in clus.get_bbas())
        ax.text(x, y, elems, ha='center', va='center', fontsize=8)
        # 簇内散度
        intra = clus.intra_divergence()
        if intra is not None:
            ax.text(x, y - size * 0.45, f"D_intra={intra:.4f}",
                    ha='center', va='top', fontsize=8)
        else:
            ax.text(x, y - size * 0.45, "D_intra=NA",
                    ha='center', va='top', fontsize=8)

    # 绘制簇间连线及距离标注
    K = len(clus_list)
    for i in range(K):
        for j in range(i + 1, K):
            x1, y1 = positions[i]
            x2, y2 = positions[j]
            ax.plot([x1, x2], [y1, y2], color='gray', linewidth=0.8)
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            dist_val = dist_mat[i][j]
            ax.text(mid_x, mid_y, f"{dist_val:.2f}", color='red', fontsize=8,
                    ha='center', va='center', bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.5))

    if save_path:
        savefig(fig, save_path)
    if show:
        plt.show()
