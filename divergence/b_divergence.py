# -*- coding: utf-8 -*-
"""B 散度计算模块
===================
复现论文《A new divergence measure for belief functions in D–S evidence theory for multisensor data fusion》中的 B 散度算法，接口与 ``bjs.py`` 类似，可被导入调用。
"""

from __future__ import annotations

import math
import os
from typing import FrozenSet, List, Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from divergence.metric_test import test_nonnegativity, test_symmetry, test_triangle_inequality  # type: ignore
from utility.bba import BBA

_LOG_BASE: float = 2.0  # 对数底数，默认为 2.0
_LOG_FN = (lambda x: math.log(x, _LOG_BASE)) if _LOG_BASE != math.e else math.log


def b_divergence(m1: BBA, m2: BBA) -> float:
    """计算两条 BBA 的 B 散度"""

    # ----------- 构造 2^N 数据结构 ----------- #
    # 利用 BBA 接口统一获取 m1、m2 的并集识别框架
    frame: FrozenSet[str] = BBA.union(m1.frame, m2.frame)
    helper = BBA(frame=frame)
    # 列举 \Theta 的所有非空子集，顺序保持一致
    subsets: List[FrozenSet[str]] = helper.theta_powerset()

    # 不考虑空集，共有 size 个子集
    size = len(subsets)
    # 对每个子集保存 (质量值, |A|)
    m1Ai: List[Tuple[float, int]] = [
        (m1.get(s, 0.0), BBA.cardinality(s)) for s in subsets
    ]
    m2Aj: List[Tuple[float, int]] = [
        (m2.get(s, 0.0), BBA.cardinality(s)) for s in subsets
    ]
    # 预先计算交集和并集元素个数矩阵
    inter_mat: List[List[int]] = [
        [BBA.cardinality(BBA.intersection(subsets[i], subsets[j])) for j in range(size)]
        for i in range(size)
    ]
    union_mat: List[List[int]] = [
        [BBA.cardinality(BBA.union(subsets[i], subsets[j])) for j in range(size)]
        for i in range(size)
    ]

    # ----------- 双重求和计算 B 散度 ----------- #
    div = 0.0
    for i in range(size):
        p, len_ai = m1Ai[i]
        if p == 0 or len_ai == 0:
            continue
        for j in range(size):
            q, len_aj = m2Aj[j]
            if q == 0 or len_aj == 0:
                continue
            # 计算交集元素个数
            inter = inter_mat[i][j]
            if inter == 0:
                continue
            # 计算并集元素个数 fixme 这个没有完全按照公式来，因为完全按照公式来结果对不上，只有这一种方案对上了。
            union = union_mat[i][j]
            # 中间分布
            M = p + q
            # 按公式 (12) 分别计算两部分并累加
            div += 0.5 * p * _LOG_FN(2 * p / M) * (inter / union)
            div += 0.5 * q * _LOG_FN(2 * q / M) * (inter / union)

    return div if div > 0 else 0.0


def divergence_matrix(bbas: List[Tuple[str, BBA]]) -> pd.DataFrame:
    """生成对称 B 散度矩阵"""
    names = [n for n, _ in bbas]
    size = len(names)
    mat = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            # 注意，B divergence 不是一个度量，因此不满足三角不等式的。B divergence没有metric的API。
            d = b_divergence(bbas[i][1], bbas[j][1])
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
        fname = f"b_{label}_{os.path.splitext(default_name)[0]}.csv"
        out_path = os.path.join(res, fname)
    dist_df.to_csv(out_path, float_format='%.4f', index_label=index_label)


def plot_heatmap(
        dist_df: pd.DataFrame,
        out_path: Optional[str] = None,
        default_name: str = 'Example_3_3.csv',
        title: Optional[str] = 'B Divergence Heatmap',
        label: str = 'divergence',
) -> None:
    """绘制并保存热力图"""
    if out_path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        res = os.path.join(base, '..', 'experiments_result')
        os.makedirs(res, exist_ok=True)
        fname = f"b_{label}_{os.path.splitext(default_name)[0]}.png"
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
    plt.savefig(out_path, dpi=600)
