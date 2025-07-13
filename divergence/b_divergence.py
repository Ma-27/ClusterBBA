# -*- coding: utf-8 -*-
"""B 散度计算模块
===================
复现论文《A new divergence measure for belief functions in D–S evidence theory for multisensor data fusion》中的 B 散度算法，接口与 ``bjs.py`` 类似，可被导入调用。
B divergence并非一个真正的度量。所以并没有提供metric接口。函数命名遵照原文，原文中命名为divergence，则函数名也为divergence。在命名规范中，只有原文中没有出现过的、符合度量公理的修改才被命名为metric。
"""

from __future__ import annotations

import math
import os
from typing import FrozenSet, List, Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from config import LOG_BASE
from utility.bba import BBA
from utility.plot_style import apply_style
from utility.plot_utils import savefig

apply_style()


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
            # 相关系数权重 ρ(A_i,A_j)   fixme 这个是完全按照原文公式来的，但是有个别数值示例与原文中的结果对不上。
            weight_p = inter / len_aj  # |Ai ∩ Aj| / |Aj|
            weight_q = inter / len_ai  # |Ai ∩ Aj| / |Ai|

            # 中间分布
            M = p + q
            # 按公式 (12) 分别计算两部分并累加
            div += 0.5 * p * math.log(2 * p / M, LOG_BASE) * weight_p
            div += 0.5 * q * math.log(2 * q / M, LOG_BASE) * weight_q

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
    savefig(fig, out_path)
