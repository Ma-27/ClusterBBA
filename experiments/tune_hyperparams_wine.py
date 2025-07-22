# -*- coding: utf-8 -*-
"""搜索最佳 ``lambda`` 与 ``mu`` 超参数
=====================================

遍历候选 ``lambda`` 和 ``mu`` 组合，调用 :func:`experiments.application_wine.evaluate_accuracy` 评估 ``Proposed`` 方法在 Wine 数据集上的分类准确率，输出 ``len(candidates)`` 个最优超参组合。提供 ``--debug`` 选项仅尝试前两组组合。
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
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
from experiments.application_wine import evaluate_accuracy


def apply_hyperparams(lambda_val: float, mu_val: float) -> None:
    """更新全局超参数供后续计算使用."""

    config.LAMBDA = lambda_val
    config.MU = mu_val

    # 同步到已导入模块中，确保所有调用处使用同一组超参
    multi_clusters.LAMBDA = lambda_val
    multi_clusters.MU = mu_val
    my_rule.LAMBDA = lambda_val
    my_rule.MU = mu_val


def search(params: Iterable[Tuple[float, float]]) -> List[Tuple[float, float, float]]:
    """遍历参数组合并按准确率降序返回 ``(lambda, mu, acc)`` 列表."""

    results: List[Tuple[float, float, float]] = []

    param_list = list(params)
    # tqdm 用于展示超参组合枚举进度
    pbar = tqdm(param_list, desc="搜索进度", ncols=PROGRESS_NCOLS)
    for lam, mu in pbar:
        # --------- 设置当前组合的超参 --------- #
        apply_hyperparams(lam, mu)

        # --------- 评估当前超参下的准确率 --------- #
        acc = evaluate_accuracy(show_progress=False, debug=False, data_progress=False)

        # 在进度条上显示当前结果
        pbar.set_postfix({"lambda": lam, "mu": mu, "acc": f"{acc:.3f}"})

        results.append((lam, mu, acc))

    # 按准确率降序排列
    results.sort(key=lambda x: x[2], reverse=True)
    return results


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="搜索最佳 lambda、mu 组合")
    parser.add_argument("--debug", action="store_true", help="仅测试前两组参数")
    args = parser.parse_args()

    # 候选的 lambda 和 mu 值，将会被展开组合为 (lambda, mu) 的形式进行遍历
    candidates = [
        0.125, 0.1428, 0.1666, 0.2, 0.25, 0.3333, 0.5, 0.8,
        1.0, 1.2, 1.5, 2, 3, 4, 5, 6, 7, 8,
    ]
    pairs = list(itertools.product(candidates, candidates))

    if args.debug:
        # 调试模式下仅取前两组参数快速验证流程
        pairs = pairs[:2]

    results = search(pairs)
    top_k = len(candidates)
    print("\nBest Params:")
    # 输出前 ``len(candidates)`` 组最优超参
    for idx, (lam, mu, acc) in enumerate(results[:top_k], start=1):
        print(f"best {idx}: lambda={lam}, mu={mu}, acc={acc:.3f}")
