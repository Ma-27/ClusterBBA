# -*- coding: utf-8 -*-
"""
Jousselme Distance Calculator Module
===================================
提供 Jousselme distance 的可导入接口，供其他脚本调用。

接口：
- jousselme_distance(m1, m2) -> float
- distance_matrix(bbas) -> pd.DataFrame
- save_csv(dist_df, out_path=None, default_name, label) -> None
- plot_heatmap(dist_df, out_path=None, default_name, title) -> None

示例：
```python
from utility.io import load_bbas
from divergence.jousselme import (
    distance_matrix,
    save_csv,
    plot_heatmap,
)
import pandas as pd

# 假设 data/examples/Example_3_3.csv 已存在
df = pd.read_csv('data/examples/Example_3_3.csv')
bbas, _ = load_bbas(df)

# 计算 Jousselme 距离矩阵
j_df = distance_matrix(bbas)
save_csv(j_df, default_name='Example_3_3.csv')
plot_heatmap(j_df, default_name='Example_3_3.csv', title='Jousselme Distance Heatmap')
```
Jousselme distance 本身已满足度量性质（非负、对称、符合三角不等式），无需额外变换。
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import FrozenSet, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from utility.bba import BBA
from utility.plot_style import apply_style
from utility.plot_utils import savefig

apply_style()


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------


def _powerset(frame: FrozenSet[str]) -> List[FrozenSet[str]]:
    """返回 ``frame`` 的所有非空子集（由 :class:`BBA` 提供）。"""
    helper = BBA(frame=frame)
    return helper.theta_powerset()


def _jaccard(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    """Jaccard 相似度 ``|A ∩ B| / |A ∪ B|``，假定 ``A``、``B`` 均非空。"""
    inter = BBA.cardinality(BBA.intersection(a, b))
    union = BBA.cardinality(BBA.union(a, b))
    return inter / union if union else 0.0


@lru_cache(maxsize=32)
def _similarity_matrix(frame: Tuple[str, ...]) -> Tuple[List[FrozenSet[str]], np.ndarray]:
    """针对给定 frame 预计算并缓存相似度矩阵 D。"""
    fset = frozenset(frame)
    subsets = _powerset(fset)
    size = len(subsets)

    # 构造指示矩阵，每一行代表一个子集在 frame 中的成员关系
    elems = list(frame)
    indicator = np.zeros((size, len(elems)), dtype=int)
    for idx, s in enumerate(subsets):
        indicator[idx] = [1 if e in s else 0 for e in elems]

    inter = indicator @ indicator.T
    card = indicator.sum(axis=1)
    union = card[:, None] + card[None, :] - inter
    union[union == 0] = 1
    mat = inter / union
    return subsets, mat


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def jousselme_distance(m1: BBA, m2: BBA) -> float:
    """计算两条 BBA 之间的 Jousselme distance。

    Parameters
    ----------
    m1, m2 : BBA
        待比较的两条基本概率分配（可包含空集，内部自动忽略空集质量）。

    Returns
    -------
    float
        Jousselme distance，理论范围 ``[0, 1]``。
    """
    # 统一识别框架
    frame = tuple(sorted(m1.frame | m2.frame))
    subsets, D = _similarity_matrix(frame)

    # 构造差分向量 Δm
    diff = np.array([m1.get(A, 0.0) - m2.get(A, 0.0) for A in subsets])

    # 0.5 * Δm^T * D * Δm
    acc = diff @ D @ diff.T
    dist = np.sqrt(0.5 * max(acc, 0.0))

    # 理论上已在 [0,1]，但数值误差下需要裁剪
    return max(0.0, min(dist, 1.0))


def distance_matrix(bbas: List[BBA]) -> pd.DataFrame:
    """生成对称 Jousselme 距离矩阵 DataFrame。"""
    names = [bba.name for bba in bbas]
    size = len(names)
    mat = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            d = jousselme_distance(bbas[i], bbas[j])
            mat[i][j] = mat[j][i] = d
    return pd.DataFrame(mat, index=names, columns=names).round(4)


def save_csv(
        dist_df: pd.DataFrame,
        out_path: Optional[str] = None,
        default_name: str = "Example_3_3.csv",
        label: str = "distance",
        index_label: str = "BBA",
) -> None:
    """保存距离矩阵为 CSV 文件，可指定输出路径或使用默认文件名。"""
    if out_path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        res = os.path.join(base, "..", "experiments_result")
        os.makedirs(res, exist_ok=True)
        fname = f"jousselme_{label}_{os.path.splitext(default_name)[0]}.csv"
        out_path = os.path.join(res, fname)
    dist_df.to_csv(out_path, float_format="%.4f", index_label=index_label)


def plot_heatmap(
        dist_df: pd.DataFrame,
        out_path: Optional[str] = None,
        default_name: str = "Example_3_3_3.csv",
        title: str = "Jousselme Distance Heatmap",
        label: str = "distance",
) -> None:
    """绘制并保存热力图。"""
    if out_path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        res = os.path.join(base, "..", "experiments_result")
        os.makedirs(res, exist_ok=True)
        fname = f"jousselme_{label}_{os.path.splitext(default_name)[0]}.png"
        out_path = os.path.join(res, fname)

    fig, ax = plt.subplots()
    cax = ax.matshow(dist_df.values)
    fig.colorbar(cax)
    ax.set_xticks(range(len(dist_df)))
    ax.set_xticklabels(dist_df.columns, rotation=90)
    ax.set_yticks(range(len(dist_df)))
    ax.set_yticklabels(dist_df.index)
    ax.set_title(title)

    savefig(fig, out_path)
