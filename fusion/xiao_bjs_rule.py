# -*- coding: utf-8 -*-
r"""
Xiao BJS-Belief Entropy 组合规则
===============================

复现 Fuyuan Xiao 在 *Information Fusion* 46 (2019) 23-32 提出的
"multi-sensor data fusion based on the belief divergence measure of evidences
and the belief entropy" 方法（章节 4.1–4.3）。

注意，这个方法并非纯用 BJS 来做出决策，而是综合了信息熵和外部 Credibility 等因素。
如果从纯 BJS 的角度算权的话，可以参考 xiao_bjs_pure_rule.py，实际上大部分实验还是使用的这个思路来对比其他人的 benchmark。

流程概述
--------
1. **距离矩阵** —— 使用 ``divergence.bjs.bjs_divergence`` 计算两两 BJS 距离，
   进而求出每条证据的平均距离 \(\tilde{\text{BJS}}_i\)。
2. **可信度 Crd** —— 令支持度 \(\text{Sup}_i = 1/\tilde{\text{BJS}}_i\)，归一化
   后得到 \(\text{Crd}_i\)。
3. **信息量 IV** —— 计算 Deng 熵 \(E_d(m_i)\)，令
   \(\text{IV}_i = \exp\bigl(E_d(m_i)\bigr)\) 并归一化得到
   \(\tilde{\text{IV}}_i\)。
4. **权重 ω** —— 调整可信度，得到
   \(\text{ACrd}_i = \text{Crd}_i\,\tilde{\text{IV}}_i\)，再归一化为最终权重
   \(\omega_i\)。
5. **加权平均证据** —— 以 \(\omega_i\) 对 BBAs 加权求平均得到 ``WAE(m)``。
6. **Dempster 融合** —— 将 ``WAE(m)`` 与自身做 ``(k-1)`` 次 Dempster 正交和，
   得到最终融合结果。

公开接口
~~~~~~~~~
- ``xiao_bjs_combine(bbas: List[BBA]) -> BBA``
- ``credibility_degrees(bbas: List[BBA]) -> List[float]``
- ``information_volume(bbas: List[BBA]) -> List[float]``
- ``_weighted_average_bba(bbas: List[BBA]) -> BBA``
"""
from __future__ import annotations

import math
from functools import reduce
from itertools import combinations
from typing import Iterable, List, Optional

import numpy as np

# 依赖本项目内现成工具函数 / 模块
from config import EPS
from divergence.bjs import bjs_divergence  # Belief JS divergence
from entropy.deng_entropy import deng_entropy
from fusion.ds_rule import combine_multiple  # Dempster ⊕ 组合
from utility.bba import BBA

__all__ = [
    "xiao_bjs_combine",
    "origin_credibility_degrees",
    "credibility_degrees",
    "information_volume",
    "_weighted_average_bba",
]


# -----------------------------------------------------------------------------
# Step 1 — BJS 距离 / 可信度
# -----------------------------------------------------------------------------

def _distance_matrix(bbas: List[BBA]) -> np.ndarray:
    """构造 ``k×k`` 的 BJS 距离矩阵。"""
    k = len(bbas)
    dmat = np.zeros((k, k))  # shape: (k, k)
    for i, j in combinations(range(k), 2):
        d = bjs_divergence(bbas[i], bbas[j])
        dmat[i, j] = dmat[j, i] = d
    return dmat


def origin_credibility_degrees(bbas: List[BBA]) -> List[float]:
    """按照 Xiao‑BJS 方法得到每条证据的 *可信度* Crd_i。"""
    if not bbas:
        raise ValueError("BBA 列表为空。")

    dmat = _distance_matrix(bbas)
    # 平均距离  \tilde{BJS}_i = Σ_j BJS_{ij} / (k-1)
    avg_dist = dmat.sum(axis=1) / (len(bbas) - 1)
    # 支持度 Sup_i = 1 / avg_dist_i
    support = 1.0 / (avg_dist + EPS)  # 避免除零
    total_sup = support.sum()
    # 可信度权重 Crd_i
    return (support / total_sup).tolist()


def credibility_degrees(bbas: List[BBA]) -> List[float]:
    """计算 Xiao-BJS 原始方法的权重 ω_i。"""
    # origin_credibility_degrees 与 information_volume 分别代表两类度量
    crd = np.array(origin_credibility_degrees(bbas))
    iv = np.array(information_volume(bbas))
    # 动态权重为二者乘积后归一化
    dyn = crd * iv
    weight = dyn / dyn.sum()
    return weight.tolist()


# -----------------------------------------------------------------------------
# Step 2 — 信息量 / 不确定度
# -----------------------------------------------------------------------------

def information_volume(bbas: List[BBA]) -> List[float]:
    """返回每条证据的信息量 \\(IV_i = e^{E_d}\\)。"""
    iv = [math.exp(deng_entropy(b)) for b in bbas]
    total = sum(iv)
    return [v / total for v in iv]


# -----------------------------------------------------------------------------
# Step 3 — 加权平均 BBA
# -----------------------------------------------------------------------------

def _weighted_average_bba(bbas: List[BBA], mu: Optional[Iterable[float]] = None,
                          nu: Optional[Iterable[float]] = None) -> BBA:
    """根据权重计算加权平均 BBA。

    ``mu`` 和 ``nu`` 分别对应论文中的充分度和重要度，用于计算
    *静态可靠度* ``w(SR_i) = mu_i × nu_i``。若未提供，则默认为 ``1``。
    """

    crd = np.array(origin_credibility_degrees(bbas))
    iv_norm = np.array(information_volume(bbas))

    dyn_rel = crd * iv_norm  # ACrd_i

    # 静态可靠度因子
    if mu is None:
        mu = [1.0] * len(bbas)
    if nu is None:
        nu = [1.0] * len(bbas)
    sta_rel = np.array(mu, dtype=float) * np.array(nu, dtype=float)

    weight = dyn_rel * sta_rel
    weight /= weight.sum()  # ω_i

    # 统一识别框架
    frame = reduce(BBA.union, (b.frame for b in bbas), frozenset())
    proto = BBA({}, frame=frame)
    mass_mat = np.array([[b.get_mass(fs) for fs in proto.keys()] for b in bbas])
    # mass_mat shape: (k, |Θ|)
    avg_values = weight @ mass_mat  # ω · m, shape: (|Θ|,)
    avg_values = np.round(avg_values, 4)  # 与论文数值示例保持一致

    total = float(avg_values.sum())
    if abs(total - 1.0) > EPS:
        # 调整最大质量项，避免因四舍五入导致的未归一
        idx = int(np.argmax(avg_values))
        avg_values[idx] += 1.0 - total

    return BBA({fs: float(v) for fs, v in zip(proto.keys(), avg_values)}, frame=frame)


# -----------------------------------------------------------------------------
# 主接口
# -----------------------------------------------------------------------------

def xiao_bjs_combine(bbas: List[BBA], mu: Optional[Iterable[float]] = None,
                     nu: Optional[Iterable[float]] = None, ) -> BBA:
    """应用 Xiao BJS-Belief Entropy 规则融合多条 BBA。"""
    if not bbas:
        raise ValueError("输入 BBA 列表为空。")
    if len(bbas) == 1:
        return bbas[0]

    wae = _weighted_average_bba(bbas, mu=mu, nu=nu)
    copies = [wae] * len(bbas)
    return combine_multiple(copies)
