# -*- coding: utf-8 -*-
"""概率分布数据结构
================

提供针对 Pignistic 概率分布的统一管理接口。

接口
----
- ``Probability`` : 概率分布数据结构，保存名称与识别框架；
- ``pignistic(bba)`` : 将 :class:`BBA` 转为概率分布；
- ``argmax(prob)`` : 返回概率最大的命题及其概率值。
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, Tuple, Optional

# 依赖本项目内现成工具函数 / 模块
from utility.bba import BBA

__all__ = ["Probability", "pignistic", "argmax"]


class Probability(dict):
    """简单的概率分布容器。"""

    frame: FrozenSet[str]
    name: str

    def __init__(self,
                 mass: Optional[Dict[FrozenSet[str], float]] = None,
                 frame: Optional[Iterable[str]] = None,
                 name: str | None = None) -> None:
        mass = mass or {}
        elems = set(frame) if frame is not None else set()
        for fs in mass.keys():
            elems.update(fs)
        self.frame = frozenset(elems)
        self.name = name or ""

        expanded: Dict[FrozenSet[str], float] = {
            frozenset(fs): float(v) for fs, v in mass.items() if fs
        }
        for e in self.frame:
            expanded.setdefault(frozenset({e}), 0.0)
        super().__init__(expanded)

    def get_prob(self, fs: Iterable[str] | FrozenSet[str]) -> float:
        """获取指定命题的概率."""
        key = frozenset(fs)
        return float(super().get(key, 0.0))

    def to_series(self, order: Iterable[str]) -> list[float]:
        """按照给定顺序生成概率列表."""
        data = {BBA.format_set(fs): self.get_prob(fs) for fs in self.keys()}
        return [data.get(col, 0.0) for col in order]


# -------------------------- Pignistic 转换 -------------------------- #

def pignistic(bba: BBA) -> Probability:
    """将 ``bba`` 转换为 Pignistic 概率分布 ``BetP``。"""

    frame = bba.frame
    result: Dict[FrozenSet[str], float] = {frozenset({e}): 0.0 for e in frame}
    total = 1.0 - bba.get_mass(frozenset())
    if total <= 0:
        return Probability(result, frame=frame, name=bba.name)

    for focal, mass in bba.items():
        if not focal or mass == 0:
            continue
        share = float(mass) / len(focal) / total
        for elem in focal:
            result[frozenset({elem})] += share

    return Probability(result, frame=frame, name=bba.name)


def argmax(prob: Probability) -> Tuple[FrozenSet[str], float]:
    """返回概率最大的命题及其概率."""

    if not prob:
        return frozenset(), 0.0

    key = max(prob.keys(), key=lambda fs: prob.get_prob(fs))
    return key, prob.get_prob(key)
