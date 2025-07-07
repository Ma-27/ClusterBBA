# -*- coding: utf-8 -*-
"""Example 1.4 RD_CCJS 与 BJS 距离演示
=================================================
在完备的识别框架 Ω={A,B} 中设定两条 BBA：
    m1(A)=α, m1(B)=1-α
    m2(A)=0.0001, m2(B)=0.9999
构造单元素簇 Clus1={m1}, Clus2={m2}，当 α∈[0,1] 变化时
计算并绘制两簇间的 RD_CCJS 距离，同时与 BJS、B 及 RB 散度对比。
该脚本亦简单验证 RD_CCJS 距离的度量性质。
"""

from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cluster.one_cluster import initialize_empty_cluster
from divergence.b_divergence import b_divergence
from divergence.bjs import bjs_divergence
from divergence.metric_test import run_all_tests
from divergence.rb_divergence import rb_divergence
from divergence.rd_ccjs import rd_ccjs_divergence, divergence_matrix
from utility.bba import BBA
# 依赖本项目内现成工具函数 / 模块
from utility.formula_labels import (
    LABEL_RD_CCJS,
    LABEL_BJS,
    LABEL_B_DIV,
    LABEL_RB_DIV,
    LABEL_ALPHA,
)
from utility.plot_style import apply_style
from utility.plot_utils import highlight_overlapping_lines, savefig

apply_style()


# ------------------------------ 核心计算 ------------------------------ #

def compute_distances(alphas: List[float]) -> pd.DataFrame:
    """返回给定 α 序列下的 RD_CCJS、BJS、B 与 RB 散度数据表"""
    records = []
    for a in alphas:
        m1 = BBA({frozenset({"A"}): a, frozenset({"B"}): 1 - a})
        m2 = BBA({frozenset({"A"}): 0.0001, frozenset({"B"}): 0.9999})

        c1 = initialize_empty_cluster("Clus1")
        c1.add_bba("m1", m1)
        c2 = initialize_empty_cluster("Clus2")
        c2.add_bba("m2", m2)

        rd = rd_ccjs_divergence(c1, c2, max(c1.h, c2.h))
        bj = bjs_divergence(m1, m2)
        b = b_divergence(m1, m2)
        rb = rb_divergence(m1, m2)
        records.append([a, rd, bj, b, rb])
    return pd.DataFrame(
        records,
        columns=["alpha", "RD_CCJS", "BJS", "B_div", "RB_div"],
    ).round(6)


# ------------------------------ 绘图函数 ------------------------------ #

def plot_rd_curve(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(df["alpha"], df["RD_CCJS"], label=LABEL_RD_CCJS)
    ax.set_xlabel(LABEL_ALPHA)
    ax.set_ylabel("Distance")
    ax.set_title(f"{LABEL_RD_CCJS} Distance with {LABEL_ALPHA} Variation")
    highlight_overlapping_lines(ax)
    savefig(fig, out_path)


def plot_compare_curve(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots()
    ax.plot(df["alpha"], df["RD_CCJS"], label=LABEL_RD_CCJS)
    ax.plot(df["alpha"], df["BJS"], label=LABEL_BJS)
    ax.plot(df["alpha"], df["B_div"], label=LABEL_B_DIV)
    ax.plot(df["alpha"], df["RB_div"], label=LABEL_RB_DIV)
    ax.set_xlabel(LABEL_ALPHA)
    ax.set_ylabel("Distance")
    ax.legend()
    ax.set_title(f"Compare {LABEL_RD_CCJS} with Other Divergences")
    highlight_overlapping_lines(ax)
    savefig(fig, out_path)


# ------------------------------ 主函数 ------------------------------ #

if __name__ == "__main__":
    alphas = list(np.linspace(0.0001, 0.9999, 101))
    df = compute_distances(alphas)

    # 保存结果到 CSV 文件
    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.normpath(os.path.join(base_dir, "..", "experiments_result"))
    os.makedirs(result_dir, exist_ok=True)

    csv_path = os.path.join(result_dir, "example_1_4_rd_ccjs.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")

    # 绘制并保存 RD_CCJS 距离曲线和对比图
    plot_rd_curve(df, os.path.join(result_dir, "example_1_4_rd_curve.png"))
    plot_compare_curve(df, os.path.join(result_dir, "example_1_4_compare.png"))

    # 度量性质简单验证，选取 alpha=0,0.5,1 构造簇
    test_alphas = [0.0, 0.5, 1.0]
    clusters = []
    for idx, a in enumerate(test_alphas, start=1):
        m = BBA({frozenset({"A"}): a, frozenset({"B"}): 1 - a})
        c = initialize_empty_cluster(f"Clus{idx}")
        c.add_bba(f"m{idx}", m)
        clusters.append(c)
    dist_df = divergence_matrix(clusters)
    print("\n----- RD_CCJS Metric Matrix -----")
    print(dist_df.to_string())
    print(run_all_tests(dist_df))
