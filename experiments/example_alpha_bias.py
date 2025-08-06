# -*- coding: utf-8 -*-
"""专家超参 \alpha 实验
======================

构造 "极端冲突" 的双簇, 观察不同 \alpha 下簇 ``Clus1``与 ``Clus2`` 的人均支持度之比 :math:`SupRate(n,\alpha)` 随克隆次数
``n`` 的变化。
"""

from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt  # 新增
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from cluster.one_cluster import initialize_empty_cluster
from divergence.bjs import divergence_matrix as bjs_matrix
from divergence.rd_ccjs import rd_ccjs_divergence
from utility.bba import BBA
from utility.io import load_bbas
from utility.plot_style import apply_style, alpha_colors
from utility.plot_utils import savefig

apply_style()


# ------------------------------ 基础 BBA ------------------------------ #


def _load_example_bbas() -> tuple[BBA, BBA]:
    """从 ``Example_1_2.csv`` 加载两条基线 BBA ``m1`` 与 ``m2``"""
    base = os.path.dirname(os.path.abspath(__file__))
    # todo 此处可修改为目标 Dataset
    csv_path = os.path.join(base, "..", "data", "examples", "Example_1_2.csv")
    csv_path = os.path.normpath(csv_path)
    df = pd.read_csv(csv_path)
    bbas, _ = load_bbas(df)
    m1 = next(b for b in bbas if b.name == "m1")
    m2 = next(b for b in bbas if b.name == "m2")
    return m1, m2


# ------------------------------ 核心计算 ------------------------------ #


def _avg_intra_distances(cluster) -> List[float]:
    """返回簇内每条 BBA 的平均 BJS 距离 ``d_{i,j}``"""
    bbas = cluster.get_bbas()
    n = len(bbas)
    if n <= 1:
        return [0.0] * n
    dist_df = bjs_matrix(bbas)
    return [float(dist_df.iloc[j].sum() / (n - 1)) for j in range(n)]


def _inter_distance(c1, c2) -> float:
    """计算两簇间 ``RD_CCJS`` 距离 ``D_i``"""
    return rd_ccjs_divergence(c1, c2, max(c1.h, c2.h))


def compute_sup_rate(alphas: List[float], clones: List[int]) -> pd.DataFrame:
    """遍历 ``n`` 和 ``\alpha`` 计算簇 ``Clus1`` 与 ``Clus2`` 的人均支持度比"""
    m1, m2 = _load_example_bbas()
    records = []
    for n in clones:
        c1 = initialize_empty_cluster("Clus1")
        for i in range(n + 1):
            clone = BBA(dict(m1), frame=m1.frame,
                        name=f"{m1.name}_{i}")
            c1.add_bba(clone, _init=True)
        c2 = initialize_empty_cluster("Clus2")
        c2.add_bba(BBA(dict(m2), frame=m2.frame, name=f"{m2.name}_0"),
                   _init=True)

        dists_c1 = _avg_intra_distances(c1)
        dists_c2 = _avg_intra_distances(c2)
        D = _inter_distance(c1, c2)
        n1, n2 = len(dists_c1), len(dists_c2)

        for a in alphas:
            sup1 = [(n1 ** a) * np.exp(-d) * np.exp(-D) for d in dists_c1]
            sup2 = [(n2 ** a) * np.exp(-d) * np.exp(-D) for d in dists_c2]
            avg1 = float(np.mean(sup1)) if sup1 else 0.0
            avg2 = float(np.mean(sup2)) if sup2 else 0.0
            ratio = avg1 / avg2 if avg2 != 0 else np.nan
            records.append([n1, a, ratio])
    return pd.DataFrame(records, columns=["n", "alpha", "sup_rate"]).round(6)


# ------------------------------ 绘图函数 ------------------------------ #


def plot_curves(df: pd.DataFrame, alphas: List[float], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)

    label_map = {
        1 / 8: r"$\alpha=\frac{1}{8}$", 1 / 7: r"$\alpha=\frac{1}{7}$",
        1 / 6: r"$\alpha=\frac{1}{6}$", 1 / 5: r"$\alpha=\frac{1}{5}$",
        1 / 4: r"$\alpha=\frac{1}{4}$", 1 / 3: r"$\alpha=\frac{1}{3}$",
        1 / 2: r"$\alpha=\frac{1}{2}$", 1: r"$\alpha=1$",
        2: r"$\alpha=2$", 3: r"$\alpha=3$", 4: r"$\alpha=4$",
        5: r"$\alpha=5$", 6: r"$\alpha=6$", 7: r"$\alpha=7$",
        8: r"$\alpha=8$"
    }

    for a in alphas:
        sub = df[np.isclose(df["alpha"], a)]
        idx = _alpha_to_color_idx(a)
        color = alpha_colors[idx]
        ax.plot(
            sub["n"],
            sub["sup_rate"],
            label=label_map[a],
            color=color,
            linewidth=1.2,
        )

    ax.set_xlabel(r"Clone number $n$")
    ax.set_ylabel(r"$SupRate(n,\alpha)$")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mtick.LogLocator(base=10))
    ax.yaxis.set_minor_locator(mtick.LogLocator(base=10, subs="all"))
    ax.yaxis.set_minor_formatter(mtick.NullFormatter())

    leg = ax.legend(
        ncol=1, fontsize=5, handlelength=1.2,
        columnspacing=0.5, frameon=True,
        loc="upper left", bbox_to_anchor=(1.02, 1.00)
    )
    leg.get_frame().set_alpha(0.3)

    # 如果还需微调，可取消上面 constrained_layout，改用手动：
    # fig.subplots_adjust(left=0.08, right=0.75, top=0.92, bottom=0.12)

    savefig(fig, out_path)


def _alpha_to_color_idx(a: float) -> int:
    # 根据 ``utility.plot_style.alpha_colors`` 中的注释索引配色
    mapping = {
        1 / 8: 2,  # α = 1/8
        1 / 7: 3,  # α = 1/7
        1 / 6: 4,  # α = 1/6
        1 / 5: 5,  # α = 1/5
        1 / 4: 6,  # α = 1/4
        1 / 3: 7,  # α = 1/3
        1 / 2: 8,  # α = 1/2
        1: 9,  # α = 1
        2: 10,  # α = 2
        3: 11,  # α = 3
        4: 12,  # α = 4
        5: 13,  # α = 5
        6: 14,  # α = 6
        7: 15,  # α = 7
        8: 16,  # α = 8
    }
    return mapping[a]


# ------------------------------ 主函数 ------------------------------ #

if __name__ == "__main__":
    # 与 ``alpha_colors`` 的注释顺序一致的 \alpha 取值
    alphas = [
        1 / 8, 1 / 7, 1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2,
        1, 2, 3, 4, 5, 6, 7, 8
    ]
    clones = list(range(10))
    # 计算 log 相对支持率
    df = compute_sup_rate(alphas, clones)

    base = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.normpath(os.path.join(base, "..", "experiments_result"))
    os.makedirs(result_dir, exist_ok=True)

    csv_path = os.path.join(result_dir, "example_alpha_bias.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")

    plot_curves(df, alphas, os.path.join(result_dir, "example_alpha_bias.png"))

    print("Alpha hyperparameter experiment completed.")
    print(f"Results saved to: {csv_path}")

