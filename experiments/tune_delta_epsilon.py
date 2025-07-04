# -*- coding: utf-8 -*-
"""RD_CCJS 超参数搜索
=======================

在 Example 1.1 的基础上，遍历若干 ``delta`` 与 ``epsilon`` 组合，
计算曲线平滑度与端点误差的综合得分，从而给出推荐的超参数。
"""

from __future__ import annotations

import itertools
import math
import os
from typing import Tuple

import numpy as np

# 依赖本项目内现成工具函数 / 模块
from experiments.example_1_1 import compute_distances

ALPHAS = list(np.linspace(0.0, 1.0, 101))


def score(delta: float, epsilon: float) -> Tuple[float, float, float]:
    """返回 (总得分, 端点偏差, 最大相邻差)."""
    df = compute_distances(ALPHAS, delta=delta, epsilon=epsilon)
    values = df["RD_CCJS"].values
    end_dev = ((values[0] - math.sqrt(2)) ** 2 + (values[-1] - math.sqrt(2)) ** 2) / 2
    smooth = np.max(np.abs(np.diff(values)))
    return end_dev * 10 + smooth, end_dev, smooth


if __name__ == "__main__":
    deltas = np.logspace(-5, -4, 3)
    epsilons = np.logspace(-6, -4, 3)

    records = []
    best = (None, float("inf"))
    for d, e in itertools.product(deltas, epsilons):
        s, dev, sm = score(d, e)
        records.append((d, e, s, dev, sm))
        if s < best[1]:
            best = ((d, e), s)

    best_delta, best_epsilon = best[0]
    print(f"Best delta={best_delta:g}, epsilon={best_epsilon:g}, score={best[1]:.6f}")

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    res_dir = os.path.join(base, "experiments_result")
    os.makedirs(res_dir, exist_ok=True)
    out_path = os.path.join(res_dir, "tune_delta_epsilon.csv")
    np.savetxt(out_path, np.array(records), delimiter=",", fmt="%g", header="delta,epsilon,score,end_dev,smooth",
               comments="")
    print(f"Results saved to: {out_path}")
