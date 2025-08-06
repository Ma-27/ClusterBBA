# -*- coding: utf-8 -*-
"""
Dempster-Shafer 组合规则模块
==============================

实现经典的 Dempster-Shafer (DS) 证据组合规则，接口设计和 ``bjs.py`` 保持一致，方便在其他脚本中直接调用。

注意：论文《Combining belief functions based on distance of evidence.pdf》的 Table 1 数值示例中，m1-m5一组的值和论文的数值设定不对应。如果 m5 的值完全等于 m4 的值，则 Table1 的数值就正确了。

接口
-----
- **ds_combine(m1, m2) -> BBA**
    两条 BBA 按 DS 规则组合后返回新的 BBA。
- **combine_multiple(bbas) -> BBA**
    依次将多条 BBA 合并为一条最终 BBA，相当于
    ``(((m1 ⊕ m2) ⊕ m3) ⊕ ... ⊕ mk)``。
  若需要保存结果，可直接使用 ``utility.io.save_bba``。

示例::

    from utility.io import load_bbas, save_bba
    from fusion.ds_rule import combine_multiple
    import pandas as pd

    df = pd.read_csv('data/examples/Example_3_3.csv')
    bbas, _ = load_bbas(df)
    result = combine_multiple([m for _, m in bbas])
    save_bba(result, default_name='Example_3_3_ds.csv')

"""
from __future__ import annotations

from typing import Dict, List, FrozenSet

# 依赖本项目内现成工具函数 / 模块
from config import EPS  # 项目内统一的极小值常数（如 1e-12）
from utility.bba import BBA

__all__ = [
    "conflict_coefficient",
    "ds_combine",
    "combine_multiple",
]


# ---------------------------------------------------------------------------
#  DS 组合规则核心实现
# ---------------------------------------------------------------------------


def conflict_coefficient(m1: BBA, m2: BBA) -> float:
    """返回两条 BBA 在 Dempster 组合下的冲突系数 ``K``。"""
    k = 0.0
    for B, p in m1.items():
        for C, q in m2.items():
            if not BBA.intersection(B, C):
                k += p * q
    return k


def ds_combine(m1: BBA, m2: BBA) -> BBA:
    """两条 BBA 的 Dempster-Shafer 正交和。

    公式::

        m(A) = [∑_{B ∩ C = A} m1(B) · m2(C)] / (1 - K)
        K    = ∑_{B ∩ C = ∅} m1(B) · m2(C)

    当 ``1 - K ≈ 0``（完全冲突）或 ``K > 1``（异常冲突）时抛出 ``ValueError``。
    """
    new_mass: Dict[FrozenSet[str], float] = {}

    for B, p in m1.items():
        for C, q in m2.items():
            inter = BBA.intersection(B, C)
            if inter:
                new_mass[inter] = new_mass.get(inter, 0.0) + p * q

    K = conflict_coefficient(m1, m2)
    if K > 1:
        raise ValueError("DS 组合失败: 冲突系数 K 大于 1")

    denom = 1.0 - K
    if denom <= EPS:
        raise ValueError("DS 组合失败: 完全冲突, 1 - K ≈ 0")

    # 归一化
    for A in new_mass:
        new_mass[A] /= denom

    # 保留原始信念框架 Θ，确保所有焦元（包括 m(A)=0）都能展开
    new_frame = BBA.union(m1.frame, m2.frame)
    return BBA(new_mass, frame=new_frame)


def combine_multiple(bbas: List[BBA]) -> BBA:
    """顺序组合多条 BBA。若列表为空则抛出 ``ValueError``。"""
    if not bbas:
        raise ValueError("BBA 列表为空，无法组合")
    result = bbas[0]
    for bba in bbas[1:]:
        result = ds_combine(result, bba)
    return result
