# -*- coding: utf-8 -*-
"""
Modified Average 组合规则
=========================

复现 Deng 等人在 2004 年提出的 *modified average* 证据融合方法。
其核心步骤为：先利用 Jousselme 距离计算每条证据的可信度权重，
据此求得加权平均 BBA，再将其与自身做 ``(n-1)`` 次 Dempster 正交和。

公开接口
---------
- ``modified_average_combine(bbas) -> BBA``
- ``credibility_degrees(bbas) -> List[float]``
- ``_weighted_average_bba(bbas) -> BBA``
"""

from __future__ import annotations

from functools import reduce
from itertools import combinations
from typing import List

import numpy as np

# 依赖本项目内现成工具函数 / 模块
from divergence.jousselme import jousselme_distance  # type: ignore
from fusion.ds_rule import combine_multiple  # 已有 Dempster 组合
from utility.bba import BBA

__all__ = [
    "modified_average_evidence",
    "credibility_degrees",
    "_weighted_average_bba",
]


def _similarity_matrix(bbas: List[BBA]) -> np.ndarray:
    """构造 ``k×k`` 相似度矩阵 ``S``，其中 ``S_ij = 1 - d(m_i, m_j)``。"""
    k = len(bbas)
    sim = np.eye(k)
    for i, j in combinations(range(k), 2):
        s = 1.0 - jousselme_distance(bbas[i], bbas[j])
        sim[i, j] = sim[j, i] = s
    return sim


def credibility_degrees(bbas: List[BBA]) -> List[float]:
    """
    按照 Deng (2004) 方法计算每条证据的权重 (可信度)。
    论文中每条证据的支持度 ``Sup(m_i)`` 为与其他证据的相似度之和，因此需要排除 ``S_ii`` 自身的相似度 ``1``。
    """
    if not bbas:
        raise ValueError("BBA 列表为空。")
    S = _similarity_matrix(bbas)
    # 排除对角线自身相似度 1
    support = S.sum(axis=1) - np.ones(len(bbas))  # Sup(m_i)
    total = support.sum()
    if total == 0:
        # 极端情况：所有相似度均为 0，此时回退到等权
        k = len(bbas)
        return [1.0 / k] * k
    return (support / total).tolist()  # Crd_i


def _weighted_average_bba(bbas: List[BBA]) -> BBA:
    """根据权重计算加权平均 BBA。"""
    weights = np.array(credibility_degrees(bbas))
    frame = reduce(BBA.union, (b.frame for b in bbas), frozenset())
    # 初始化所有子集质量为 0
    proto = BBA({}, frame=frame)
    mass_mat = np.array([[b.get_mass(fs) for fs in proto.keys()] for b in bbas])
    # 累加权重 * 质量
    avg_values = weights @ mass_mat  # shape [2^Theta,]
    avg_mass = {fs: float(v) for fs, v in zip(proto.keys(), avg_values)}
    return BBA(avg_mass, frame=frame)


def modified_average_evidence(bbas: List[BBA]) -> BBA:
    """
    主流程：
    ① 计算权重并生成加权平均 BBA；  
    ② 将该平均 BBA 与自身做 (n‑1) 次 Dempster 正交和。
    """
    if not bbas:
        raise ValueError("输入 BBA 列表为空。")
    if len(bbas) == 1:
        return bbas[0]

    avg = _weighted_average_bba(bbas)
    copies = [avg] * len(bbas)
    return combine_multiple(copies)
