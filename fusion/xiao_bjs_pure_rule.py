# -*- coding: utf-8 -*-
"""
纯 BJS 组合规则
===============

根据 BJS 距离计算支持度与可信度, 不涉及 Deng 熵或其他信息量调节。
流程与 :mod:`fusion.xiao_rb_rule` 类似, 仅将距离替换为 BJS。

公开接口
---------
- ``xiao_bjs_pure_combine(bbas: List[BBA]) -> BBA``
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
from divergence.bjs import bjs_divergence
from fusion.ds_rule import combine_multiple
from utility.bba import BBA

__all__ = [
    "xiao_bjs_pure_combine",
    "credibility_degrees",
    "_weighted_average_bba",
]


# -----------------------------------------------------------------------------
# Step 1 — BJS 距离 / 可信度
# -----------------------------------------------------------------------------

def _distance_matrix(bbas: List[BBA]) -> np.ndarray:
    """构造 ``k×k`` 的 BJS 距离矩阵。"""
    k = len(bbas)
    dmat = np.zeros((k, k))
    for i, j in combinations(range(k), 2):
        d = bjs_divergence(bbas[i], bbas[j])
        dmat[i, j] = dmat[j, i] = d
    return dmat


def credibility_degrees(bbas: List[BBA]) -> List[float]:
    """按照纯 BJS 方法计算可信度权重。"""
    if not bbas:
        raise ValueError("BBA 列表为空。")

    if len(bbas) == 1:
        return [1.0]

    dmat = _distance_matrix(bbas)
    avg_dist = dmat.sum(axis=1) / (len(bbas) - 1)
    support = 1.0 / (avg_dist + EPS)  # 避免除零
    total_sup = support.sum()
    return (support / total_sup).tolist()


# -----------------------------------------------------------------------------
# Step 2 — 加权平均 BBA
# -----------------------------------------------------------------------------

def _weighted_average_bba(bbas: List[BBA]) -> BBA:
    """根据可信度计算加权平均 BBA。"""
    weights = np.array(credibility_degrees(bbas))

    frame = reduce(BBA.union, (b.frame for b in bbas), frozenset())
    proto = BBA({}, frame=frame)
    mass_mat = np.array([[b.get_mass(fs) for fs in proto.keys()] for b in bbas])
    avg_values = weights @ mass_mat
    avg_values = np.round(avg_values, 4)

    return BBA({fs: float(v) for fs, v in zip(proto.keys(), avg_values)})


# -----------------------------------------------------------------------------
# Step 3 — 主接口
# -----------------------------------------------------------------------------

def xiao_bjs_pure_combine(bbas: List[BBA]) -> BBA:
    """多传感器数据融合主流程。"""
    if not bbas:
        raise ValueError("BBA 列表为空。")

    if len(bbas) == 1:
        return bbas[0]

    wae = _weighted_average_bba(bbas)
    copies = [wae] * len(bbas)
    return combine_multiple(copies)
