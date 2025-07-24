# -*- coding: utf-8 -*-
"""基于 kfold 的 ``lambda``、``mu`` 网格调参
=====================================

遍历每个折测试集的候选 ``lambda`` 与 ``mu`` 组合，选择分类准确率最高的一组参数。若出现并列，会在 ``alpha`` 取 ``0.5``、``1.0``、``1.5`` 时计算平均准确率以打破平局。结果按折号保存到 ``experiments_result`` 目录。
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional, Callable

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
import pandas as pd
from utility.io_application import load_application_dataset_cv

# 各数据集的准确率评估函数
from experiments.tune_hyperparams_iris import (
    evaluate_accuracy as eval_iris,
)
from experiments.tune_hyperparams_seeds import (
    evaluate_accuracy as eval_seeds,
)
from experiments.tune_hyperparams_glass import (
    evaluate_accuracy as eval_glass,
)
from experiments.tune_hyperparams_wine import (
    evaluate_accuracy as eval_wine,
)

EVAL_FUNCS: Dict[str, Callable[..., float]] = {
    "iris": eval_iris,
    "seeds": eval_seeds,
    "glass": eval_glass,
    "wine": eval_wine,
}


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


def collect_accuracy(samples: List[Tuple[int, List, str, int]], lambda_val: float, mu_val: float,
                     eval_func: Callable[..., float], *, alpha: float | None = None) -> float:
    """在给定样本上计算分类准确率."""

    if alpha is not None:
        config.ALPHA = alpha
        try:
            my_rule.ALPHA = alpha
        except Exception:
            pass

    # --------- 设置当前组合的超参 --------- #
    apply_hyperparams(lambda_val, mu_val)

    # 去掉 fold 信息以复用现有评估函数
    simple_samples = [(idx, bbas, gt) for idx, bbas, gt, _ in samples]

    return float(
        eval_func(
            samples=simple_samples, show_progress=False, data_progress=False, debug=False
        )
    )


# ---------------------------------------------------------------------------
# 主搜索逻辑
# ---------------------------------------------------------------------------

def search_fold(samples: List[Tuple[int, List, str, int]], params: Iterable[Tuple[float, float]],
                eval_func: Callable[..., float]) -> Tuple[float, float]:
    """在单个折上搜索使准确率最高的 ``lambda``、``mu``."""

    best_pairs: List[Tuple[float, float]] = []
    best_acc = -1.0

    # 进度条显示当前尝试的超参组
    pbar = tqdm(list(params), desc="搜索进度", ncols=PROGRESS_NCOLS)
    for lam, mu in pbar:
        acc = collect_accuracy(samples, lam, mu, eval_func)
        pbar.set_postfix({"lambda": lam, "mu": mu, "acc": f"{acc:.4f}"})
        if acc > best_acc + 1e-6:
            best_acc = acc
            best_pairs = [(lam, mu)]
        elif abs(acc - best_acc) <= 1e-6:
            best_pairs.append((lam, mu))

    pbar.close()

    if len(best_pairs) == 1:
        return best_pairs[0]

    # 平局时在多个 alpha 上比较平均准确率，并展示进度
    alphas = [0.5, 1.0, 1.5]
    best_pair = best_pairs[0]
    best_avg = -1.0
    tie_pbar = tqdm(best_pairs, desc="平局处理", ncols=PROGRESS_NCOLS)
    for lam, mu in tie_pbar:
        scores = [collect_accuracy(samples, lam, mu, eval_func, alpha=a) for a in alphas]
        avg = sum(scores) / len(scores)
        tie_pbar.set_postfix({"lambda": lam, "mu": mu, "avg": f"{avg:.4f}"})
        if avg > best_avg + 1e-6:
            best_avg = avg
            best_pair = (lam, mu)
    tie_pbar.close()

    return best_pair


def tune_kfold_params(dataset: str, *, csv_path: str | Path | None = None, debug: bool = False,
                      fold: Optional[int] = None) -> List[Dict[str, float]]:
    """搜索并保存每折最优的 ``lambda``、``mu`` 组合.

    Parameters
    ----------
    dataset : str
        数据集名称，如 ``"iris"``、``"seeds"`` 等。
    csv_path : str or Path, optional
        数据集 CSV 路径，默认为 ``data/kfold_xu_bba_<dataset>.csv``。
    debug : bool, optional
        调试模式，仅测试前两组参数。
    fold : int, optional
        仅评估指定折号，默认为 ``None`` 表示遍历所有折。
    """

    if csv_path is None:
        csv_path = Path(__file__).resolve().parents[1] / "data" / f"kfold_xu_bba_{dataset}.csv"
    csv_path = Path(csv_path)

    samples = load_application_dataset_cv(csv_path=csv_path, debug=False)

    if dataset not in EVAL_FUNCS:
        raise ValueError(f"未知的数据集: {dataset}")
    eval_func = EVAL_FUNCS[dataset]

    # 候选的 lambda 和 mu 值，将会被展开组合为 (lambda, mu) 的形式进行遍历
    candidates = [
        0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3333, 0.4,
        0.5, 0.6, 0.8, 1.0, 1.25, 1.6667, 2.0, 2.5, 3.0, 5.0,
        10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0,
    ]

    # len(candidates) = 27  -> 27**2 = 729 pairs
    pairs = list(itertools.product(candidates, candidates))
    if debug:
        pairs = pairs[:2]

    folds = sorted(set(f for *_, f in samples))
    if fold is not None:
        if fold not in folds:
            raise ValueError(f"fold={fold} 超出范围 {folds}")
        folds = [fold]
    results: List[Dict[str, float]] = []
    out_path = Path(__file__).resolve().parents[1] / "experiments_result" / f"kfold_best_params_{dataset}.csv"

    for fold in folds:
        fold_samples = [s for s in samples if s[3] == fold]
        lam_best, mu_best = search_fold(fold_samples, pairs, eval_func)
        results.append({"fold": fold, "lambda": lam_best, "mu": mu_best})
        pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8")
        print(f"Fold {fold}: lambda={lam_best}, mu={mu_best}")

    return results


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="搜索 K 折交叉验证 BBA 的最佳 lambda、mu 组合"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="iris",
        help="数据集名称, 对应 kfold_xu_bba_<dataset>.csv",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="仅评估指定折号 (从 0 开始), 默认为全部折",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式, 只尝试前两组超参",
    )
    args = parser.parse_args()

    tune_kfold_params(args.dataset, debug=args.debug, fold=args.fold)
