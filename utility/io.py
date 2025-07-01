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

__all__ = [
    "load_bbas",
    "parse_focal_set",
    "format_set",
]


def parse_focal_set(cell: str) -> FrozenSet[str]:
    """
    解析列名，去除花括号并按“∪”分割生成元素集，如 {A ∪ B} -> frozenset({'A', 'B'})

    支持格式示例
    ------------
    - ``"{A ∪ B}"``  → ``frozenset({'A', 'B'})``
    - ``"A ∪ B"``    → ``frozenset({'A', 'B'})``
    - ``"A"``        → ``frozenset({'A'})``
    - ``"∅"`` / ``"{}"`` / ``""`` → ``frozenset()``
    """
    cell = cell.strip()
    if not cell or cell in {"∅", "{}"}:
        return frozenset()

    if cell.startswith("{") and cell.endswith("}"):
        cell = cell[1:-1]

    items = [e.strip() for e in cell.split("∪") if e.strip()]
    return frozenset(items)


def format_set(fs: FrozenSet[str]) -> str:
    """
    将 *frozenset* 反格式化为字符串（空集 → ``"∅"``；其余按字母升序并用 ``" ∪ "`` 连接）。

    如 frozenset({'A', 'B'}) -> {A ∪ B}
    """
    if not fs:
        return "∅"
    return "{" + " ∪ ".join(sorted(fs)) + "}"


def _row_to_bba(row: pd.Series, focal_cols: List[str]) -> Dict[FrozenSet[str], float]:
    """内部助手：把一行 DataFrame 转成 *{focal_set: mass}* 字典。"""
    bba: Dict[FrozenSet[str], float] = {}
    for col in focal_cols:
        bba[parse_focal_set(col)] = float(row[col])
    return bba


def load_bbas(df: pd.DataFrame, ) -> Tuple[List[Tuple[str, Dict[FrozenSet[str], float]]], List[str]]:
    """
    读取 *DataFrame* 并返回 **[(名称, bba_dict), …]** 及焦元列顺序。

    Parameters
    ----------
    df : pd.DataFrame
        至少包含一列 `'BBA'` 作为标签，其余列为焦元质量。

    Returns
    -------
    list
        ``[(name, {focal: mass, …}), …]``  （这就是一条 BBA 的数据结构）
    list
        焦元列标题顺序，用于后续 DataFrame 重建保持对齐。
    """

    # 提取列名（跳过 'BBA' 列）作为焦元表示
    focal_cols = [c for c in df.columns if c != "BBA"]
    bbas: List[Tuple[str, Dict[FrozenSet[str], float]]] = []  # List[Tuple[name, bba_dict]]
    for _, row in df.iterrows():
        name = str(row.get("BBA", "m"))
        # 将列名转焦元集合，行值转质量
        mass = {parse_focal_set(col): float(row[col]) for col in focal_cols}
        bbas.append((name, mass))  # 存储名称与 BBA
    return bbas, focal_cols
