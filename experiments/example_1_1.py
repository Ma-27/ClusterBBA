# -*- coding: utf-8 -*-
"""Example 1.1 RD_CCJS 簇冲突实验
================================
根据 `研究思路构建` 中对簇-簇冲突的定义，
本脚本在两条随 α 变化的 BBA 之间计算 `RD_CCJS`、`BJS` 距离，
观察簇间冲突随 α 的变化情况，并与 Dempster 冲突系数对比。
"""

from __future__ import annotations

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cluster.one_cluster import initialize_empty_cluster
from divergence.bjs import bjs_metric
from divergence.rd_ccjs import rd_ccjs_metric


# ------------------------------ 核心计算 ------------------------------ #

def compute_distances(alphas: List[float]) -> pd.DataFrame:
    """返回给定 α 序列下的 RD_CCJS、Dempster 冲突系数与 BJS 距离"""
    records = []
    for a in alphas:
        m1 = {frozenset({"A1"}): a, frozenset({"A2"}): 1 - a}
        m2 = {frozenset({"A1"}): 1 - a, frozenset({"A2"}): a}

        c1 = initialize_empty_cluster("Clus1")
        c1.add_bba("m1", m1)
        c2 = initialize_empty_cluster("Clus2")
        c2.add_bba("m2", m2)

        H = max(c1.h, c2.h)
        rd = rd_ccjs_metric(c1, c2, H)
        bj = bjs_metric(m1, m2)
        k = a * a + (1 - a) * (1 - a)  # Dempster 冲突系数
        records.append([a, rd, bj, k])
    return pd.DataFrame(records, columns=["alpha", "RD_CCJS", "BJS", "K"]).round(6)


# ------------------------------ 绘图函数 ------------------------------ #

def plot_curves(df: pd.DataFrame, out_path: str) -> None:
    plt.figure()
    plt.plot(df["alpha"], df["RD_CCJS"], label="RD_CCJS")
    plt.plot(df["alpha"], df["BJS"], label="BJS")
    plt.plot(df["alpha"], df["K"], label="K")
    plt.xlabel("alpha")
    plt.ylabel("value")
    plt.legend()
    plt.title("Example 1.1 Cluster Conflict")
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()


# ------------------------------ 主函数 ------------------------------ #

if __name__ == "__main__":
    # todo 在这里更改 alpha 的范围和种子数目
    alphas = list(np.linspace(0.0, 1.0, 101))
    df = compute_distances(alphas)

    # 保存结果到 CSV
    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.normpath(os.path.join(base_dir, "..", "experiments_result"))
    os.makedirs(result_dir, exist_ok=True)

    csv_path = os.path.join(result_dir, "example_1_1_rd_ccjs.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")

    # 绘制并保存曲线图
    plot_curves(df, os.path.join(result_dir, "example_1_1_curve.png"))
    print("Example 1.1 draw curves to depict conflicting degrees of each measurement.")
    print(f"Example 1.1 results saved to:{csv_path}")
