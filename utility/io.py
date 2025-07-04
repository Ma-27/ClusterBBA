# -*- coding: utf-8 -*-
"""
utility.io
==========

专门收纳针对 **BBA (Basic Belief Assignment)** 数据的常用 I/O 工具函数，避免在
项目各脚本里重复实现同一逻辑。

导出接口
--------
- parse_focal_set(cell: str) -> FrozenSet[str]
- format_set(fs: FrozenSet[str]) -> str
- load_bbas(df: pd.DataFrame)
      读取 *DataFrame*，返回 [(name, bba_dict), …] 及焦元列顺序

约定
----
1. **标签列名固定为 `'BBA'`**；其余列全部视为焦元质量。
2. 焦元列标题形如 `'{A ∪ B}'`、`'A'`、`'∅'` 等均可；内部统一转
   `frozenset({'A','B'})` 或空集合。
3. 空集字符串统一采用 `'∅'`；若确需使用 `'{}'` 亦可自动识别。

"""

from typing import Dict, FrozenSet, List, Tuple

import pandas as pd

from utility.bba import BBA

__all__ = [
    "load_bbas",
    "parse_focal_set",
    "format_set",
]


def parse_focal_set(cell: str) -> FrozenSet[str]:
    """包装 :func:`BBA.parse_focal_set`，保持旧接口。"""
    return BBA.parse_focal_set(cell)


def format_set(fs: FrozenSet[str]) -> str:
    """包装 :func:`BBA.format_set`，保持旧接口。"""
    return BBA.format_set(fs)


def _row_to_bba(row: pd.Series, focal_cols: List[str]) -> BBA:
    """内部助手：把一行 DataFrame 转成 :class:`BBA` 对象。"""
    mass: Dict[FrozenSet[str], float] = {}
    for col in focal_cols:
        mass[parse_focal_set(col)] = float(row[col])
    return BBA(mass)


def load_bbas(df: pd.DataFrame, ) -> Tuple[List[Tuple[str, BBA]], List[str]]:
    """
    读取 *DataFrame* 并返回 **[(名称, bba_dict), …]** 及焦元列顺序。

    Parameters
    ----------
    df : pd.DataFrame
        至少包含一列 `'BBA'` 作为标签，其余列为焦元质量。

    Returns
    -------
    list
        ``[(name, BBA), …]``  （返回名称与 :class:`BBA` 对象列表）
    list
        焦元列标题顺序，用于后续 DataFrame 重建保持对齐。
    """

    # 提取列名（跳过 'BBA' 列）作为焦元表示
    focal_cols = [c for c in df.columns if c != "BBA"]
    bbas: List[Tuple[str, BBA]] = []
    for _, row in df.iterrows():
        name = str(row.get("BBA", "m"))
        # 将列名转焦元集合，行值转质量
        mass = {parse_focal_set(col): float(row[col]) for col in focal_cols}
        bbas.append((name, BBA(mass)))  # 存储名称与 BBA
    return bbas, focal_cols
