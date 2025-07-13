# -*- coding: utf-8 -*-
"""
Murphy 平均组合规则（Simple Average + (n-1) 次 Dempster 正交和）
参考：Murphy, C. K. “Combining belief functions when evidence conflicts.” Decision Support Systems 29 (2000) 1-9.
"""
from __future__ import annotations

from functools import reduce
from typing import List

# 依赖本项目内现成工具函数 / 模块
from fusion.ds_rule import combine_multiple  # 已有的 DS 证据融合和实现
from mean.mean_bba import compute_avg_bba  # type: ignore
from utility.bba import BBA

__all__ = ["murphy_combine", "_average_bba", "credibility_degrees"]


def _average_bba(bbas: List[BBA]) -> BBA:
    """按 Murphy 论文的定义计算 ``n`` 条 BBA 的简单平均。由于每条 BBA 的质量和为 1，平均后仍满足归一化。"""
    if not bbas:
        raise ValueError("BBA 列表为空。")

    avg = compute_avg_bba(bbas)
    # 保留原始帧，确保与 ds_rule 接口兼容
    frame = reduce(BBA.union, (b.frame for b in bbas), frozenset())
    return BBA(dict(avg), frame=frame)


def murphy_combine(bbas: List[BBA]) -> BBA:
    """
    Murphy 组合主流程：
    ① 先求简单平均 \bar m；
    ② 再把 \bar m 与自身做 (n − 1) 次 Dempster 正交和。
    """
    if not bbas:
        raise ValueError("输入 BBA 列表为空。")
    if len(bbas) == 1:
        return bbas[0]

    avg = _average_bba(bbas)
    copies = [avg] * len(bbas)
    return combine_multiple(copies)


def credibility_degrees(bbas: List[BBA]) -> List[float]:
    """
    **仅用于打印演示**。Murphy 原文并未使用“可信度”权重，
    这里返回等权 1/n，方便在测试脚本里保留原有输出格式。
    """
    n = len(bbas)
    return [1.0 / n] * n
