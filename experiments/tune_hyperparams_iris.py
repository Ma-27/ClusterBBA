# -*- coding: utf-8 -*-
"""在 Iris 数据集上搜索 ``lambda`` 与 ``mu`` 超参
============================================

遍历候选 ``lambda`` 和 ``mu`` 组合, 调用 :func:`experiments.application_iris.evaluate_accuracy` 评估分类准确率, 输出前 ``len(candidates)`` 组最优参数.  提供 ``--debug`` 选项可仅尝试前两组参数以快速验证流程.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from functools import cmp_to_key
from typing import Iterable, Tuple, List

from tqdm import tqdm

# 确保包导入路径指向项目根目录
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 依赖本项目内现成工具函数 / 模块
import config
from config import PROGRESS_NCOLS
import cluster.multi_clusters as multi_clusters
import fusion.my_rule as my_rule
from experiments.application_iris import evaluate_accuracy


# ---------------------------------------------------------------------------
# 参数同步与评估逻辑
# ---------------------------------------------------------------------------


def apply_hyperparams(lambda_val: float, mu_val: float) -> None:
    """同步 ``lambda`` 与 ``mu`` 超参, 确保所有模块读取到相同值."""

    config.LAMBDA = lambda_val
    config.MU = mu_val

    multi_clusters.LAMBDA = lambda_val
    multi_clusters.MU = mu_val
    my_rule.LAMBDA = lambda_val
    my_rule.MU = mu_val


def search(params: Iterable[Tuple[float, float]]) -> List[Tuple[float, float, float]]:
    """遍历参数组合并按准确率降序返回 ``(lambda, mu, acc)`` 列表."""

    results: List[Tuple[float, float, float]] = []
    param_list = list(params)

    # 进度条显示当前尝试的超参组
    pbar = tqdm(param_list, desc="搜索进度", ncols=PROGRESS_NCOLS)
    for lam, mu in pbar:
        # --------- 设置当前组合的超参 --------- #
        apply_hyperparams(lam, mu)

        # --------- 评估当前超参下的准确率 --------- #
        # 评估当前超参组合在随机顺序下的平均准确率
        acc = evaluate_accuracy(show_progress=False, debug=False, data_progress=False)

        # 在进度条上显示当前结果
        pbar.set_postfix({"lambda": lam, "mu": mu, "acc": f"{acc:.6f}"})

        results.append((lam, mu, acc))

    # 按准确率降序排列，若准确率相同则选择更接近 1 的结果
    def _cmp(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> int:
        if abs(a[2] - b[2]) <= 1e-6:
            diff_a = abs(a[2] - 1.0)
            diff_b = abs(b[2] - 1.0)
            if diff_a < diff_b:
                return -1
            if diff_a > diff_b:
                return 1
            return 0
        return -1 if a[2] > b[2] else 1

    results.sort(key=cmp_to_key(_cmp))
    return results


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="搜索 Iris 数据集上的最佳 lambda、mu")
    # 指定是否运行在调试模式下
    parser.add_argument("--debug", action="store_true", help="仅测试前两组参数")
    args = parser.parse_args()

    # 候选的 lambda 和 mu 值，将会被展开组合为 (lambda, mu) 的形式进行遍历
    candidates = [
        0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3333, 0.4,
        0.5, 0.6, 0.8, 1.0, 1.25, 1.6667, 2.0, 2.5, 3.0, 5.0,
        10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0
    ]

    # len(candidates) = 27  -> 27**2 = 729 pairs
    pairs = list(itertools.product(candidates, candidates))

    if args.debug:
        # 调试模式下只搜索两组参数验证流程
        pairs = pairs[:2]

    results = search(pairs)
    top_k = len(candidates)
    print("\nBest Params:")
    # 输出前 ``len(candidates)`` 组最优超参
    for idx, (lam, mu, acc) in enumerate(results[:top_k], start=1):
        print(f"best {idx}: lambda={lam}, mu={mu}, acc={acc:.6f}")
