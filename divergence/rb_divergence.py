# -*- coding: utf-8 -*-
"""RB Divergence Calculation Module
=================================
复现论文提出的 Reinforced Belief (RB) 散度。依赖 ``b_divergence`` 实现。
"""

import math
import os
from typing import Dict, FrozenSet, List, Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from divergence.b_divergence import b_divergence  # type: ignore


def rb_divergence(m1: Dict[FrozenSet[str], float], m2: Dict[FrozenSet[str], float]) -> float:
    """计算两条 BBA 的 RB 散度"""
    b11 = b_divergence(m1, m1)
    b22 = b_divergence(m2, m2)
    b12 = b_divergence(m1, m2)
    rb = abs(b11 + b22 - 2 * b12) / 2
    return math.sqrt(max(0.0, min(rb, 1.0)))


def divergence_matrix(bbas: List[Tuple[str, Dict[FrozenSet[str], float]]]) -> pd.DataFrame:
    """生成 RB 散度矩阵"""
    names = [n for n, _ in bbas]
    size = len(names)
    mat = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            # 注意，RB divergence 遵照原文，不给出metric的API，直接调用这个函数就足够了。
            d = rb_divergence(bbas[i][1], bbas[j][1])
            mat[i][j] = mat[j][i] = d
    return pd.DataFrame(mat, index=names, columns=names).round(4)


# ---------------------------------------------------------------------------
# Convenience: CSV / visualisation
# ---------------------------------------------------------------------------


def save_csv(
        dist_df: pd.DataFrame,
        out_path: Optional[str] = None,
        default_name: str = 'Example_3_3.csv',
        label: str = 'divergence',
        index_label: str = 'BBA',
) -> None:
    """保存矩阵为 CSV"""
    if out_path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        res = os.path.join(base, '..', 'experiments_result')
        os.makedirs(res, exist_ok=True)
        fname = f"rb_{label}_{os.path.splitext(default_name)[0]}.csv"
        out_path = os.path.join(res, fname)
    dist_df.to_csv(out_path, float_format='%.4f', index_label=index_label)


def plot_heatmap(
        dist_df: pd.DataFrame,
        out_path: Optional[str] = None,
        default_name: str = 'Example_3_3.csv',
        title: Optional[str] = 'RB Divergence Heatmap',
        label: str = 'divergence',
) -> None:
    """绘制并保存热力图"""
    if out_path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        res = os.path.join(base, '..', 'experiments_result')
        os.makedirs(res, exist_ok=True)
        fname = f"rb_{label}_{os.path.splitext(default_name)[0]}.png"
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
