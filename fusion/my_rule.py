# -*- coding: utf-8 -*-
r"""Proposed cluster-based fusion rule
====================================

根据  ``融合折扣计算与融合算法`` 的流程实现的证据融合方法。
主要思想是先对输入 BBA 进行动态分簇，随后依据簇规模、簇内平均 BJS 距离以及簇间``RD_CCJS`` 距离计算每条证据的权重，最终生成加权平均证据并按 DS 规则融合。

流程概述
--------
1. **动态分簇** —— 按顺序加入 BBA 构建多个簇；
2. **权重计算** —— 对每条证据计算支持度 ``Sup``，并归一化为 ``Crd``；
3. **加权平均** —— 以 ``Crd`` 对证据质量值求平均得到 ``\overline m``；
4. **DS 组合** —— 将 ``\overline m`` 与自身做 ``(n-1)`` 次 Dempster 组合。

公开接口
---------
- ``my_combine(bbas: List[BBA], names: List[str] | None = None) -> BBA``
- ``credibility_degrees(bbas: List[BBA], names: List[str] | None = None) -> List[float]``
- ``_weighted_average_bba(bbas: List[BBA], names: List[str] | None = None) -> BBA``
"""
from __future__ import annotations

import math
from functools import reduce
from typing import Dict, List

import numpy as np

# 依赖本项目内现成工具函数 / 模块
from cluster.multi_clusters import construct_clusters_by_sequence  # type: ignore
from config import ALPHA, LAMBDA, MU
from divergence.bjs import divergence_matrix as bjs_matrix
from divergence.rd_ccjs import divergence_matrix as rdcc_matrix  # type: ignore
from fusion.ds_rule import combine_multiple
from utility.bba import BBA

__all__ = [
    "my_combine",
    "credibility_degrees",
    "_weighted_average_bba",
]


# ---------------------------------------------------------------------------
# Step 1 — 权重计算
# ---------------------------------------------------------------------------


def _cluster_intra_dist(clus_bbas: List[BBA]) -> Dict[str, float]:
    """返回簇内每条 BBA 的平均 BJS 距离。"""
    if len(clus_bbas) < 2:
        return {b.name: 0.0 for b in clus_bbas}

    # ``divergence_matrix`` 会生成一个对称矩阵，其中对角线为 0
    df = bjs_matrix(clus_bbas)
    result: Dict[str, float] = {}
    for idx, b in enumerate(clus_bbas):
        result[b.name] = float(df.iloc[idx].sum() / (len(clus_bbas) - 1))
    return result


def credibility_degrees(bbas: List[BBA], names: List[str] | None = None) -> List[float]:
    """按照 Proposed 的公式计算每条证据的权重。

    Parameters
    ----------
    bbas : List[BBA]
        输入的证据列表。
    names : List[str] | None, optional
        与 ``bbas`` 对应的名称列表，若为 ``None`` 则按 ``m1``、``m2`` … 顺序生成。
    """
    if not bbas:
        raise ValueError("BBA 列表为空。")

    if names is not None and len(names) != len(bbas):
        raise ValueError("names 与 bbas 长度不一致")

    # 动态分簇，返回 MultiClusters 对象
    mc = construct_clusters_by_sequence(bbas, debug=False)
    clusters = list(mc._clusters.values())

    # 簇间平均 RD_CCJS 距离
    D_i_map: Dict[str, float] = {}
    if len(clusters) >= 2:
        # ``divergence_matrix`` 返回簇-簇散度矩阵
        df_rd = rdcc_matrix(clusters)
        for idx, clus in enumerate(clusters):
            # 第 idx 行的平均值即簇 ``Clus_i`` 的平均散度 D_i
            D_i_map[clus.name] = float(df_rd.iloc[idx].sum() / (len(clusters) - 1))
    else:
        # 只有一个簇时，D_i 皆视为 0
        only = clusters[0]
        D_i_map[only.name] = 0.0

    Sup: Dict[str, float] = {}
    for clus in clusters:
        names_bbas = clus.get_bbas()
        n_i = len(names_bbas)
        intra = _cluster_intra_dist(names_bbas)
        # 该簇与其他簇的平均散度
        D_i = D_i_map.get(clus.name, 0.0)
        for b in names_bbas:
            d_ij = intra.get(b.name, 0.0)
            sup = (n_i ** ALPHA) * math.exp(-LAMBDA * d_ij) * math.exp(-MU * D_i)
            Sup[b.name] = sup

    # 归一化得到 Crd_{i,j}
    total = sum(Sup.values())
    order = names if names is not None else [b.name for b in bbas]
    weights = [Sup[n] / total for n in order]
    return weights


# ---------------------------------------------------------------------------
# Step 2 — 加权平均 BBA
# ---------------------------------------------------------------------------

def _weighted_average_bba(bbas: List[BBA], names: List[str] | None = None) -> BBA:
    """根据权重计算加权平均 BBA。"""
    weights = np.array(credibility_degrees(bbas, names))
    # 合并所有信念框架，确保每条 BBA 都能在同一 Θ 下表示
    frame = reduce(BBA.union, (b.frame for b in bbas), frozenset())
    proto = BBA({}, frame=frame)
    # 构造质量矩阵，行为证据，列为焦元
    mass_mat = np.array([[b.get_mass(fs) for fs in proto.keys()] for b in bbas])
    avg_values = weights @ mass_mat
    return BBA({fs: float(v) for fs, v in zip(proto.keys(), avg_values)}, frame=frame)


# ---------------------------------------------------------------------------
# Step 3 — 主接口
# ---------------------------------------------------------------------------

def my_combine(bbas: List[BBA], names: List[str] | None = None) -> BBA:
    """多传感器融合主流程。"""
    if not bbas:
        raise ValueError("BBA 列表为空。")

    if len(bbas) == 1:
        return bbas[0]

    # Step 1 & 2: 计算权重并求加权平均证据
    avg = _weighted_average_bba(bbas, names)
    # Step 3: 重复 DS 组合 (n-1) 次
    copies = [avg] * len(bbas)
    return combine_multiple(copies)
