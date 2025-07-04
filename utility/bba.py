# -*- coding: utf-8 -*-
"""BBA 数据结构
================

统一管理项目中的 Basic Belief Assignment (BBA)。
本类继承自 ``dict``，键为 ``frozenset``，值为 ``float``。 初始化时会自动展开到识別框架 ``\\Theta`` 的全部非空子集，缺失的焦元质量默认填充为 0.0。提供常用的焦元和集合运算接口。
"""

from __future__ import annotations

import itertools
from typing import Dict, FrozenSet, Iterable, List, Optional

__all__ = ["BBA"]


class BBA(dict):
    """单条基本概率分配。"""

    frame: FrozenSet[str]

    def __init__(self,
                 mass: Optional[Dict[FrozenSet[str], float]] = None,
                 frame: Optional[Iterable[str]] = None) -> None:
        mass = mass or {}
        elems = set(frame) if frame is not None else set()
        for fs in mass.keys():
            elems.update(fs)
        self.frame = frozenset(elems)

        expanded: Dict[FrozenSet[str], float] = {
            fs: float(v) for fs, v in mass.items() if fs
        }
        # 保留空集质量（若存在）
        if frozenset() in mass:
            expanded[frozenset()] = float(mass[frozenset()])

        for r in range(1, len(self.frame) + 1):
            for combo in itertools.combinations(sorted(self.frame), r):
                fs = frozenset(combo)
                if fs not in expanded:
                    expanded[fs] = 0.0

        super().__init__(expanded)

    # ------------------------ 常用接口 ------------------------ #
    def get_mass(self, fs: Iterable[str] | FrozenSet[str]) -> float:
        """获取指定焦元的质量值，若不存在则返回 0.0。"""
        key = frozenset(fs)
        return float(super().get(key, 0.0))

    def focal_sets(self) -> List[FrozenSet[str]]:
        """返回所有质量非零的焦元（不含空集）。"""
        return [fs for fs, m in self.items() if fs and m != 0]

    @staticmethod
    def subset_cardinality(fs: Iterable[str] | FrozenSet[str]) -> int:
        """返回集合所有非空子集的数量 ``2^|A|-1``。"""
        size = len(fs)
        return (2 ** size) - 1

    @property
    def theta_cardinality(self) -> int:
        """识别框架 ``\\Theta`` 的非空子集数量。"""
        return self.subset_cardinality(self.frame)

    def __repr__(self) -> str:  # pragma: no cover - 调试信息
        return f"BBA({dict.__repr__(self)})"

    # ------------------------ I/O 助手 ------------------------ #
    @staticmethod
    def parse_focal_set(cell: str) -> FrozenSet[str]:
        """解析集合字符串为 ``frozenset``。支持 ``"{A ∪ B}"`` 等格式。"""
        cell = cell.strip()
        if not cell or cell in {"∅", "{}"}:
            return frozenset()
        if cell.startswith("{") and cell.endswith("}"):
            cell = cell[1:-1]
        items = [e.strip() for e in cell.split("∪") if e.strip()]
        return frozenset(items)

    @staticmethod
    def format_set(fs: FrozenSet[str]) -> str:
        """将 ``frozenset`` 格式化为字符串，例如 ``{'A', 'B'}`` -> ``"{A ∪ B}"``"""
        if not fs:
            return "∅"
        return "{" + " ∪ ".join(sorted(fs)) + "}"

    def to_formatted_dict(self) -> Dict[str, float]:
        """按字符串焦元返回 ``{str: mass}`` 字典，便于构造 DataFrame。"""
        return {self.format_set(fs): float(m) for fs, m in self.items()}

    def to_series(self, order: List[str]) -> List[float]:
        """根据给定顺序生成质量值列表，不存在的焦元补 0。"""
        data = self.to_formatted_dict()
        return [data.get(col, 0.0) for col in order]

    # ------------------------ 基本属性 ------------------------ #
    def total_mass(self) -> float:
        """返回质量总和。"""
        return sum(float(m) for m in self.values())

    def validate(self, tol: float = 1e-6) -> bool:
        """检查质量和是否为 1，误差容忍度 ``tol``。"""
        return abs(self.total_mass() - 1.0) <= tol

    # ------------------------ 集合辅助 ------------------------ #
    @property
    def theta_size(self) -> int:
        """识别框架 ``\\Theta`` 元素个数。"""
        return len(self.frame)

    def theta_powerset(self, include_empty: bool = False) -> List[FrozenSet[str]]:
        """列举 ``\\Theta`` 的所有子集。``include_empty`` 控制是否包含空集。"""
        elems = sorted(self.frame)
        start = 0 if include_empty else 1
        pset: List[FrozenSet[str]] = []
        if include_empty:
            pset.append(frozenset())
        for r in range(start, len(elems) + 1):
            for c in itertools.combinations(elems, r):
                pset.append(frozenset(c))
        return pset

    @staticmethod
    def union(fs1: Iterable[str], fs2: Iterable[str]) -> FrozenSet[str]:
        """返回两个集合的并集。"""
        return frozenset(fs1).union(fs2)

    @staticmethod
    def intersection(fs1: Iterable[str], fs2: Iterable[str]) -> FrozenSet[str]:
        """返回两个集合的交集。"""
        return frozenset(fs1).intersection(fs2)

    @staticmethod
    def is_subset(a: Iterable[str], b: Iterable[str]) -> bool:
        """判断 ``a`` 是否为 ``b`` 的子集。"""
        return frozenset(a).issubset(b)

    @staticmethod
    def is_superset(a: Iterable[str], b: Iterable[str]) -> bool:
        """判断 ``a`` 是否为 ``b`` 的父集。"""
        return frozenset(a).issuperset(b)

    @staticmethod
    def cardinality(fs: Iterable[str]) -> int:
        """返回集合元素个数。"""
        return len(set(fs))

    def supersets(self, fs: Iterable[str], proper: bool = False) -> List[FrozenSet[str]]:
        """枚举 ``\\Theta`` 内含有 ``fs`` 的所有集合。"""
        base = frozenset(fs)
        return [s for s in self.theta_powerset() if base.issubset(s) and (not proper or s != base)]

    def subsets_of(self, fs: Iterable[str], proper: bool = False, include_empty: bool = False) -> List[FrozenSet[str]]:
        """枚举 ``fs`` 的所有子集，限定在 ``\\Theta`` 内。"""
        base = frozenset(fs)
        elems = [e for e in base if e in self.frame]
        res: List[FrozenSet[str]] = []
        start = 0 if include_empty else 1
        if include_empty:
            res.append(frozenset())
        for r in range(start, len(elems) + 1):
            for c in itertools.combinations(elems, r):
                sub = frozenset(c)
                if proper and sub == base:
                    continue
                res.append(sub)
        return res
