# -*- coding: utf-8 -*-
"""基于贝叶斯优化的 ``lambda``、``mu`` 超参调节
=================================================

使用 Optuna 贝叶斯优化搜索 ``lambda`` 与 ``mu`` 组合, 以最大化分类准确率并减少搜索开销。既支持在 ``kfold`` 数据上逐折评估, 亦可在完整数据集上直接搜索。若出现并列, 依照固定的 ``alpha`` 序列重新评估并比较曲线稳定性来打破平局。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
from optuna.samplers import TPESampler
from tqdm import tqdm

# 抑制 Optuna 的冗余日志, 避免干扰进度条显示
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
from utility.io_application import (
    load_application_dataset,
    load_application_dataset_cv,
)

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

ALPHA_SEQ = [
    1 / 8, 1 / 7, 1 / 6, 1 / 5, 1 / 4, 1 / 3, 1 / 2, 1,
    2, 3, 4, 5, 6, 7, 8,
]
_TIE_EPS = 1e-6


# ---------------------------------------------------------------------------
# 参数同步与评估逻辑
# ---------------------------------------------------------------------------

def apply_hyperparams(lambda_val: float, mu_val: float) -> None:
    """同步 ``lambda`` 与 ``mu`` 超参, 确保所有模块读取到相同值."""

    # --------- 更新全局配置中的超参 --------- #
    config.LAMBDA = lambda_val
    config.MU = mu_val

    # --------- 同步到其他已导入模块 --------- #
    # 这些模块在运行时读取模块级变量, 因此需要手动覆盖
    multi_clusters.LAMBDA = lambda_val
    multi_clusters.MU = mu_val
    my_rule.LAMBDA = lambda_val
    my_rule.MU = mu_val


def collect_accuracy(samples: List[Tuple[int, List, str, int]], lambda_val: float, mu_val: float,
                     eval_func: Callable[..., float], *, alpha: float | None = None, ) -> float:
    """在给定样本上计算分类准确率."""

    # 若指定 ``alpha``，临时覆盖全局的 α 超参数
    if alpha is not None:
        config.ALPHA = alpha
        try:
            my_rule.ALPHA = alpha  # 某些评估函数直接使用 ``my_rule`` 中的 α
        except Exception:
            pass

    # 应用当前的 ``lambda``、``mu`` 组合
    apply_hyperparams(lambda_val, mu_val)

    # 评估函数只接受 ``(idx, bbas, gt)`` 形式的样本
    simple_samples = [(idx, bbas, gt) for idx, bbas, gt, _ in samples]

    # 调用对应数据集的准确率评估函数
    return float(eval_func(samples=simple_samples, show_progress=False, data_progress=False, debug=False))


# ---------------------------------------------------------------------------
# 搜索逻辑
# ---------------------------------------------------------------------------

def _objective(trial: optuna.Trial, samples: List[Tuple[int, List, str, int]],
               eval_func: Callable[..., float], ) -> float:
    """Optuna 目标函数: 采样 ``lambda`` 与 ``mu`` 并返回准确率."""

    # 在对数空间中随机采样 ``lambda`` 与 ``mu``，确保覆盖宽泛范围
    lam = trial.suggest_float("lambda", 0.001, 1000.0, log=True)
    mu = trial.suggest_float("mu", 0.001, 1000.0, log=True)

    # 目标是最大化默认 ``alpha=1`` 下的准确率
    return collect_accuracy(samples, lam, mu, eval_func, alpha=1.0)


def search_fold(samples: List[Tuple[int, List, str, int]], eval_func: Callable[..., float], *, n_trials: int = 50, ) -> \
        Tuple[float, float]:
    """在单个折上使用贝叶斯优化搜索 ``lambda``、``mu``."""

    # --------- 初始化贝叶斯优化流程 --------- #
    sampler = TPESampler(seed=42)  # 使用 TPE 采样器, 结果可复现
    study = optuna.create_study(direction="maximize", sampler=sampler)
    pbar = tqdm(total=n_trials, desc="搜索进度", ncols=PROGRESS_NCOLS)

    # 回调函数: 每次 trial 完成后更新进度条显示当前结果
    def _callback(study: optuna.Study, trial: optuna.Trial) -> None:  # pragma: no cover
        pbar.update(1)
        pbar.set_postfix({
            "lambda": trial.params.get("lambda", float("nan")),
            "mu": trial.params.get("mu", float("nan")),
            "acc": f"{trial.value:.4f}",
        })

    # 运行贝叶斯优化迭代
    study.optimize(
        lambda trial: _objective(trial, samples, eval_func),
        n_trials=n_trials,
        callbacks=[_callback],
    )
    pbar.close()

    # --------- 挑选并列的最优解 --------- #
    best_acc = study.best_value
    candidates = [
        (t.params["lambda"], t.params["mu"])
        for t in study.trials
        if abs(t.value - best_acc) <= _TIE_EPS
    ]

    # 若仅有一个候选, 直接返回
    if len(candidates) == 1:
        return candidates[0]

    # --------- 依据固定 α 序列重新评估以打破平局 --------- #
    tie_results = []
    tie_pbar = tqdm(candidates, desc="平局处理", ncols=PROGRESS_NCOLS)
    for lam, mu in tie_pbar:
        # 对每个候选组合遍历给定的 α 序列, 记录对应的准确率曲线
        scores = [
            collect_accuracy(samples, lam, mu, eval_func, alpha=a) for a in ALPHA_SEQ
        ]
        tie_results.append((lam, mu, scores))
        tie_pbar.set_postfix({"lambda": lam, "mu": mu, "best": f"{max(scores):.4f}"})
    tie_pbar.close()

    # 先比较各组合在 α 序列中取得的最高准确率
    best_alpha = max(max(scores) for _, _, scores in tie_results)
    candidates = [
        (lam, mu, scores)
        for lam, mu, scores in tie_results
        if abs(max(scores) - best_alpha) <= _TIE_EPS
    ]
    if len(candidates) == 1:
        lam, mu, _ = candidates[0]
        return lam, mu

    # 仍然平局时, 计算曲线的“平稳度”作为最终判据
    final_scores = []
    for lam, mu, scores in candidates:
        arr = np.asarray(scores, dtype=float)
        # 幅度越小且方差越大, 得分越低; 反之越高
        score = (arr.max() - arr.min()) / (arr.std() + 1e-6)
        final_scores.append((score, lam, mu))

    final_scores.sort(reverse=True)
    _, lam_best, mu_best = final_scores[0]
    return lam_best, mu_best


def tune_params(dataset: str, *, csv_path: str | Path | None = None, debug: bool = False, fold: Optional[int] = None,
                n_trials: int = 50, kfold: bool = False, ) -> List[Dict[str, float]]:
    """搜索并保存最优的 ``lambda``、``mu`` 组合.

    Parameters
    ----------
    dataset : str
        数据集名称.
    kfold : bool, optional
        ``True`` 时按 kfold 分折评估, 否则在完整数据集上评估 (默认).
    """

    # --------- 准备数据与评估函数 --------- #
    if csv_path is None:
        csv_path = Path(__file__).resolve().parents[1] / "data" / f"kfold_xu_bba_{dataset}.csv"
    csv_path = Path(csv_path)

    if dataset not in EVAL_FUNCS:
        raise ValueError(f"未知的数据集: {dataset}")
    eval_func = EVAL_FUNCS[dataset]

    # --------- 在完整数据集上搜索 --------- #
    if not kfold:
        # 1. 加载完整的、未分折的数据集
        samples_simple = load_application_dataset(csv_path=csv_path, debug=False)
        # 2. 添加一个虚拟的 fold 标记, 以复用 ``search_fold``
        #    原始样本格式为 (idx, bbas, gt)，这里添加一个虚拟的折号 '0'
        #    变为 (idx, bbas, gt, 0)，从而可以复用为 k-fold 设计的 search_fold 函数
        samples = [(i, b, g, 0) for i, b, g in samples_simple]
        # 3. 如果启用了调试模式，则将优化迭代次数减少到最多 2 次，以便快速验证流程
        if debug:
            n_trials = min(2, n_trials)
        # 4. 调用核心搜索函数，在整个数据集上进行贝叶斯优化，找到最佳的 lambda 和 mu
        lam_best, mu_best = search_fold(samples, eval_func, n_trials=n_trials)
        # 5. 将找到的最佳参数存入字典，并将结果四舍五入到小数点后 4 位
        result = {"lambda": round(lam_best, 4), "mu": round(mu_best, 4)}
        # 6. 构建结果文件的输出路径
        out_path = (
                Path(__file__).resolve().parents[1]
                / "experiments_result"
                / f"bayes_best_params_full_{dataset}.csv"
        )
        pd.DataFrame([result]).to_csv(out_path, index=False, encoding="utf-8", float_format="%.4f")
        print(f"Full: lambda={lam_best:.4f}, mu={mu_best:.4f}")
        return [result]

    # --------- 按 kfold 分折搜索 --------- #

    # 1. 加载完整的、未分折的数据集
    samples = load_application_dataset_cv(csv_path=csv_path, debug=False)

    # 所有存在的折号, 供后续遍历
    folds = sorted(set(f for *_, f in samples))
    # 检查是否指定了只运行某一单折
    if fold is not None:
        # 验证输入折号是否存在于数据集中
        if fold not in folds:
            raise ValueError(f"fold={fold} 超出范围 {folds}")
        folds = [fold]

    # 若启用了调试模式
    if debug:
        # 将贝叶斯优化的迭代次数减少到一个很小的值（这里是2次），以便快速完成测试
        n_trials = min(2, n_trials)

    # 初始化一个空列表，用于存储每一折的最优参数结果
    results: List[Dict[str, float]] = []
    # 构建结果输出路径
    out_path = (
            Path(__file__).resolve().parents[1]
            / "experiments_result"
            / f"bayes_best_params_kfold_{dataset}.csv"
    )

    # 遍历需要处理的每一折 (folds 列表可能是全部折，也可能是用户指定的单折)
    for f in folds:
        # 取出只属于当前折 `f` 的样本并搜索最优参数
        fold_samples = [s for s in samples if s[3] == f]
        # 调用 search_fold 函数，在当前折的样本数据上执行贝叶斯优化，找到最佳的 lambda 和 mu
        lam_best, mu_best = search_fold(fold_samples, eval_func, n_trials=n_trials)
        # 将当前折的结果（折号、最佳lambda、最佳mu）追加到 results 列表中，数值四舍五入到小数点后4位
        results.append({"fold": f, "lambda": round(lam_best, 4), "mu": round(mu_best, 4)})
        # 在每完成一折的搜索后，都将当前的全部结果实时写入到 CSV 文件中，并打印
        pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8", float_format="%.4f")
        print(f"Fold {f}: lambda={lam_best:.4f}, mu={mu_best:.4f}")

    return results


def tune_kfold_params(*args, **kwargs) -> List[Dict[str, float]]:
    """兼容旧接口, 等同于 :func:`tune_params`."""

    return tune_params(*args, **kwargs)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="使用贝叶斯优化搜索 BBA 的最佳 lambda、mu 组合")
    # todo 指定数据集
    parser.add_argument("--dataset", type=str, default="iris", help="数据集名称, 对应 kfold_xu_bba_<dataset>.csv", )
    # todo 指定是否启用 K 折交叉验证模式
    parser.add_argument("--kfold", action="store_true", help="启用 k-fold 交叉验证模式 (默认为在完整数据集上评估)", )
    # 指定是否仅评估某一折
    parser.add_argument("--fold", type=int, default=None, help="仅评估指定折号 (从 0 开始), 默认为全部折", )
    # 指定贝叶斯优化的迭代次数
    parser.add_argument("--trials", type=int, default=50, help="贝叶斯优化迭代次数", )
    # 指定是否运行在调试模式下
    parser.add_argument("--debug", action="store_true", help="调试模式, 仅运行少量迭代", )
    args = parser.parse_args()

    # 进行调参
    tune_params(args.dataset, debug=args.debug, fold=args.fold, n_trials=args.trials, kfold=args.kfold, )
