# -*- coding: utf-8 -*-
"""
Xiao RB ‑ Multisensor Fusion Rule
================================

复现 Fuyuan Xiao 在 *Information Sciences* 514 (2020) 462‑483 第 5 章提出的
“multisensor data fusion algorithm based on the **reinforced belief (RB) divergence**”。

流程概述
--------
1. **RB 距离矩阵** — 使用 ``divergence.rb_divergence.rb_divergence`` 计算两两 RB 距离，
   得到 ``m×m`` 距离矩阵 ``M``，并计算每条证据的平均距离 \\(\tilde{RB}_i\\)。
2. **支持度 Sup** — 令支持度 \\(S_i = 1/\tilde{RB}_i\\)。
3. **可信度 Crd** — 归一化支持度得到 \\(C_i\\)，即本文算法中的最终权重。
4. **加权平均证据** — 以 \\(C_i\\) 对 BBAs 加权求平均得到 ``WAE(m)``。
5. **Dempster 融合** — 将 ``WAE(m)`` 与自身做 ``(k-1)`` 次 Dempster 正交和，得到融合结果。

公开接口
~~~~~~~~~
- ``xiao_rb_combine(bbas: List[BBA]) -> BBA``
- ``credibility_degrees(bbas: List[BBA]) -> List[float]``
- ``_weighted_average_bba(bbas: List[BBA]) -> BBA``
"""
from __future__ import annotations

from functools import reduce
from itertools import combinations
from typing import List

import numpy as np

# 依赖本项目内现成工具函数 / 模块
from config import EPS
from divergence.rb_divergence import rb_divergence  # Reinforced Belief divergence
from fusion.ds_rule import combine_multiple  # Dempster ⊕ 组合
from utility.bba import BBA

__all__ = [
    "xiao_rb_combine",
    "credibility_degrees",
    "_weighted_average_bba",
]


# -----------------------------------------------------------------------------
# Step 1 — RB 距离 / 可信度
# -----------------------------------------------------------------------------

def _distance_matrix(bbas: List[BBA]) -> np.ndarray:
    """构造 ``k×k`` 的 RB 距离矩阵。"""
    k = len(bbas)
    dmat = np.zeros((k, k))
    for i, j in combinations(range(k), 2):
        d = rb_divergence(bbas[i], bbas[j])
        dmat[i, j] = dmat[j, i] = d
    return dmat


def credibility_degrees(bbas: List[BBA]) -> List[float]:
    """按照 Xiao‑RB 方法得到每条证据的 *可信度* C_i。"""
    if not bbas:
        raise ValueError("BBA 列表为空。")

    if len(bbas) == 1:
        return [1.0]

    dmat = _distance_matrix(bbas)
    # 平均距离  \tilde{RB}_i = Σ_j RB_{ij} / (k-1)
    avg_dist = dmat.sum(axis=1) / (len(bbas) - 1)
    # 支持度  S_i = 1 / \tilde{RB}_i
    support = 1.0 / (avg_dist + EPS)  # 避免除零
    total_sup = support.sum()
    # 可信度 (权重)  C_i
    return (support / total_sup).tolist()


# -----------------------------------------------------------------------------
# Step 2 — 加权平均 BBA
# -----------------------------------------------------------------------------

def _weighted_average_bba(bbas: List[BBA]) -> BBA:
    """根据 C_i 计算加权平均 BBA。"""
    weights = np.array(credibility_degrees(bbas))

    # 统一识别框架
    frame = reduce(BBA.union, (b.frame for b in bbas), frozenset())
    proto = BBA({}, frame=frame)

    mass_mat = np.array([[b.get_mass(fs) for fs in proto.keys()] for b in bbas])
    avg_values = weights @ mass_mat  # shape: (|Θ|,)
    avg_values = np.round(avg_values, 4)  # 与论文示例对齐

    total = float(avg_values.sum())
    if abs(total - 1.0) > EPS:
        # 调整最大质量项，避免因四舍五入导致的未归一
        idx = int(np.argmax(avg_values))
        avg_values[idx] += 1.0 - total

    return BBA({fs: float(v) for fs, v in zip(proto.keys(), avg_values)})


# -----------------------------------------------------------------------------
# Step 3 — 主接口
# -----------------------------------------------------------------------------

def xiao_rb_combine(bbas: List[BBA]) -> BBA:
    """多传感器数据融合主流程。"""
    if not bbas:
        raise ValueError("BBA 列表为空。")

    if len(bbas) == 1:
        return bbas[0]

    wae = _weighted_average_bba(bbas)  # Step 5

    copies = [wae] * len(bbas)
    return combine_multiple(copies)
