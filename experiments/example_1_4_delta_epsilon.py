# -*- coding: utf-8 -*-
r"""Example 1.4 超参数敏感性实验，探索不同 ``delta`` 与 ``epsilon`` 对 RD_CCJS 的影响，绘制图表。
================================

在 :mod:`experiments.example_1_4` 的基础上，本脚本进一步分析
``RD_CCJS`` 距离对权重平滑参数 ``delta`` 与 ``epsilon`` 的敏感性。

我们仍在识别框架 :math:`\Omega=\{A,B\}` 下构造两条 BBA：
    ``m1(A)=\alpha, m1(B)=1-\alpha``
    ``m2(A)=0.0001, m2(B)=0.9999``

通过枚举 ``delta`` 或 ``epsilon`` 的不同取值，绘制三维曲面观察
``RD_CCJS`` 随 ``\alpha`` 与超参数变化的趋势。
"""

from __future__ import annotations

import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from cluster.one_cluster import initialize_empty_cluster
from divergence.rd_ccjs import rd_ccjs_metric
from utility.bba import BBA


def _calc_rd(
        alphas: Iterable[float],
        param_values: Iterable[float],
        *,
        vary_delta: bool = True,
        epsilon: float = 1e-2,
        delta: float = 1e-4,
) -> pd.DataFrame:
    """内部助手：计算 RD_CCJS 与超参数的关系"""

    records = []
    for p in param_values:
        for a in alphas:
            m1 = BBA({frozenset({"A"}): a, frozenset({"B"}): 1 - a})
            m2 = BBA({frozenset({"A"}): 0.0001, frozenset({"B"}): 0.9999})

            c1 = initialize_empty_cluster("Clus1")
            c1.add_bba("m1", m1)
            c2 = initialize_empty_cluster("Clus2")
            c2.add_bba("m2", m2)

            if vary_delta:
                rd = rd_ccjs_metric(c1, c2, max(c1.h, c2.h), delta=p, epsilon=epsilon)
                records.append([a, p, rd])
            else:
                rd = rd_ccjs_metric(c1, c2, max(c1.h, c2.h), delta=delta, epsilon=p)
                records.append([a, p, rd])

    col = "delta" if vary_delta else "epsilon"
    return pd.DataFrame(records, columns=["alpha", col, "RD_CCJS"]).round(6)


def plot_surface(df: pd.DataFrame, var: str, out_path: str) -> None:
    """绘制 RD_CCJS 随 ``alpha`` 与 ``var`` 变化的曲面，并统一坐标原点"""

    pivot = df.pivot(index=var, columns="alpha", values="RD_CCJS")
    X, Y = np.meshgrid(pivot.columns.astype(float), pivot.index.astype(float))
    Z = pivot.values

    # 绘制三维曲面图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis")
    ax.set_xlabel("alpha")
    ax.set_ylabel(var)
    ax.set_zlabel("RD_CCJS")

    # 设定坐标轴范围均从 0 开始，确保三维图共享同一原点
    ax.set_xlim(0, float(pivot.columns.astype(float).max()))
    ax.set_ylim(0, float(pivot.index.astype(float).max()))
    ax.set_zlim(0, float(np.nanmax(Z)))

    # 设置视角，仰角xx°，方位角xx°
    ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()


if __name__ == "__main__":
    alphas = np.linspace(0.0, 1.0, 101)
    deltas = np.logspace(-6, -2, 9)
    epsilons = np.logspace(-3, 0, 8)

    df_delta = _calc_rd(alphas, deltas, vary_delta=True, epsilon=1e-2)
    df_epsilon = _calc_rd(alphas, epsilons, vary_delta=False, delta=1e-4)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.normpath(os.path.join(base_dir, "..", "experiments_result"))
    os.makedirs(res_dir, exist_ok=True)

    df_delta.to_csv(os.path.join(res_dir, "example_1_4_delta.csv"), index=False)
    df_epsilon.to_csv(os.path.join(res_dir, "example_1_4_epsilon.csv"), index=False)

    # 绘制 delta 的曲面图
    plot_surface(df_delta, "delta", os.path.join(res_dir, "example_1_4_delta.png"))
    # 绘制 epsilon 的曲面图
    plot_surface(
        df_epsilon,
        "epsilon",
        os.path.join(res_dir, "example_1_4_epsilon.png"),
    )

    print("RD_CCJS sensitivity experiment finished.")
