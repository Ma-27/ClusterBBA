# -*- coding: utf-8 -*-
"""基于 5 折交叉验证的 lambda、mu 自动调节
=======================================

先根据典型衰减比例计算闭式初值, 随后采用 SPSA 在交叉验证上进行极少量微调.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from sklearn.metrics import accuracy_score

# 确保包导入路径指向项目根目录
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 依赖本项目内现成工具函数 / 模块
import config
import fusion.my_rule as my_rule
import cluster.multi_clusters as multi_clusters
from utility.io_application import load_application_dataset_cv
from utility.probability import pignistic, argmax

# 默认超参微调迭代次数
ITERATIONS = 5

# ``lambda``/``mu`` 可能的扰动幅度, 与网格搜索保持一致
DELTA_CHOICES = [
    0.125, 0.1428, 0.1666, 0.2, 0.25, 0.3333, 0.5, 0.8,
    1.0, 1.2, 1.5, 2, 3, 4, 5, 6, 7, 8,
]


def apply_hyperparams(lambda_val: float, mu_val: float) -> None:
    """同步全局超参数以供其他模块使用."""

    config.LAMBDA = lambda_val
    config.MU = mu_val
    multi_clusters.LAMBDA = lambda_val
    multi_clusters.MU = mu_val
    my_rule.LAMBDA = lambda_val
    my_rule.MU = mu_val


def evaluate_fold(samples: List[Tuple[int, List, str]],
                  combine_func: Callable,
                  label_map: Dict[str, str],
                  *,
                  lambda_val: float,
                  mu_val: float) -> Tuple[float, float]:
    """在单个折上计算准确率与平均负对数似然。

    参数
    ----
    lambda_val : float
        ``\\lambda`` 超参，影响簇内散度权重。
    mu_val : float
        ``\\mu`` 超参，影响簇间散度权重。
    """

    y_true: List[str] = []
    y_pred: List[str] = []
    nll = 0.0

    for _, bbas, gt in samples:
        # 明确传入当前待评估的超参, 避免依赖全局状态
        fused = combine_func(bbas, lambda_val=lambda_val, mu_val=mu_val)
        prob = pignistic(fused)
        fs, _ = argmax(prob)
        pred = next(iter(fs)) if fs else ""
        pred_full = label_map.get(pred, pred)
        y_true.append(gt)
        y_pred.append(pred_full)
        p_true = prob.get_prob({gt})
        p_true = max(p_true, config.EPS)
        nll -= math.log(p_true)

    acc = accuracy_score(y_true, y_pred)
    nll /= len(samples)
    return acc, nll


def evaluate_cv(samples: List[Tuple[int, List, str, int]],
                lambda_val: float,
                mu_val: float,
                combine_func: Callable,
                label_map: Dict[str, str]) -> Tuple[float, float]:
    """在所有折上评估给定超参."""

    apply_hyperparams(lambda_val, mu_val)

    folds: Dict[int, List[Tuple[int, List, str]]] = {}
    for idx, bbas, gt, fold in samples:
        folds.setdefault(fold, []).append((idx, bbas, gt))

    accs = []
    nlls = []
    for f in sorted(folds.keys()):
        # 将待评估的超参显式传入 evaluate_fold
        acc, nll = evaluate_fold(
            folds[f],
            my_rule.my_combine,
            label_map,
            lambda_val=lambda_val,
            mu_val=mu_val,
        )
        accs.append(acc)
        nlls.append(nll)
    return float(sum(accs) / len(accs)), float(sum(nlls) / len(nlls))


def spsa_update(lam: float, mu: float, samples,
                label_map: Dict[str, str],
                a: float = 0.1) -> Tuple[float, float]:
    """执行一次 SPSA 更新以微调 ``lambda`` 与 ``mu``。"""

    # 随机选择扰动幅度, 范围与网格搜索一致
    c = random.choice(DELTA_CHOICES)

    # SPSA 随机扰动两个方向上的梯度估计
    delta_lam = random.choice([-1.0, 1.0]) * c
    delta_mu = random.choice([-1.0, 1.0]) * c

    _, loss_plus = evaluate_cv(samples, lam + delta_lam, mu + delta_mu, my_rule.my_combine, label_map)
    _, loss_minus = evaluate_cv(samples, lam - delta_lam, mu - delta_mu, my_rule.my_combine, label_map)

    # 估计损失函数在兩個方向上的梯度
    g_lam = (loss_plus - loss_minus) / (2 * delta_lam)
    g_mu = (loss_plus - loss_minus) / (2 * delta_mu)

    lam_new = lam - a * g_lam
    mu_new = mu - a * g_mu

    # 将更新结果限制在合理区间 [0.125, 8]
    lam_new = min(max(lam_new, DELTA_CHOICES[0]), DELTA_CHOICES[-1])
    mu_new = min(max(mu_new, DELTA_CHOICES[0]), DELTA_CHOICES[-1])

    # 返回更新后的 ``lambda`` 和 ``mu``
    return lam_new, mu_new


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="交叉验证调节 lambda、mu")
    parser.add_argument("--csv", type=str, default=None, help="数据集 CSV 路径")
    parser.add_argument("--iterations", type=int, default=ITERATIONS, help="微调迭代次数")
    args = parser.parse_args()

    csv_path = args.csv
    if csv_path is None:
        csv_path = Path(__file__).resolve().parents[1] / "data" / "kfold_xu_bba_iris.csv"

    # ----------------------- 数据加载与标签映射 ----------------------- #
    # 加载带有 ``fold`` 列的数据集以便交叉验证
    samples = load_application_dataset_cv(csv_path=csv_path, debug=False)
    df = None
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
    except Exception:
        pass

    # 部分数据集标签可能使用缩写，此处尝试构造缩写到完整标签的映射
    label_map: Dict[str, str] = {}
    if df is not None:
        bba_cols = [c for c in df.columns if c.startswith('{') and '∪' not in c]
        abbrs = [c.strip('{} ') for c in bba_cols]
        for ab in abbrs:
            if ab.startswith('C') and ab[1:].isdigit():
                label_map[ab] = f"Class {ab[1:]}"
            else:
                for gt in df['ground_truth'].unique():
                    if gt.lower().startswith(ab.lower()):
                        label_map[ab] = gt
                        break

    # Step-1: 按典型衰减比例计算 ``lambda`` 与 ``mu`` 的闭式初值
    lam = math.log(-math.log(0.1)) / math.log(2)
    mu = math.log(-math.log(0.1)) / math.log(2)

    print(f"Initial lambda={lam:.3f}, mu={mu:.3f}")

    # 根据设定的迭代次数执行 SPSA 微调
    for i in range(args.iterations):
        lam, mu = spsa_update(lam, mu, samples, label_map)
        acc, loss = evaluate_cv(samples, lam, mu, my_rule.my_combine, label_map)
        print(
            f"Iter {i + 1}: lambda={lam:.3f}, mu={mu:.3f}, acc={acc:.4f}, loss={loss:.4f}"
        )

    # 打印最终调节后的超参
    print(f"\nTuned lambda={lam:.3f}, mu={mu:.3f}")
