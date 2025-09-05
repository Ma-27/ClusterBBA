# -*- coding: utf-8 -*-
"""基于贝叶斯优化的 ``lambda``、``mu`` (可选 ``alpha``) 超参调节
====================================================================

使用 Optuna 的 TPE 采样器搜索 ``lambda`` 与 ``mu`` 组合, 以最大化分类准确率并减少搜索开销。既支持在 ``kfold`` 数据上逐折评估, 亦可在完整数据集上直接搜索。若启用 ``alpha`` 选项, 则 ``lambda``、``mu`` 与 ``alpha`` 会被同时优化。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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

def _objective(trial: optuna.Trial, samples: List[Tuple[int, List, str, int]], eval_func: Callable[..., float], *,
               opt_alpha: bool = False, lambda_range: Tuple[float, float] = (0.001, 1000.0),
               mu_range: Tuple[float, float] = (0.001, 1000.0), ) -> float:
    """Optuna 目标函数: 采样 ``lambda``、``mu`` (及可选 ``alpha``) 并返回准确率."""

    # 在指定的对数空间内随机采样 ``lambda`` 与 ``mu``
    lam = trial.suggest_float("lambda", lambda_range[0], lambda_range[1], log=True)
    mu = trial.suggest_float("mu", mu_range[0], mu_range[1], log=True)

    if opt_alpha:
        alpha = trial.suggest_float("alpha", 1 / 8, 8.0, log=True)
        return collect_accuracy(samples, lam, mu, eval_func, alpha=alpha)

    # 目标是最大化默认 ``alpha=1`` 下的准确率
    return collect_accuracy(samples, lam, mu, eval_func, alpha=1.0)


def search_fold(samples: List[Tuple[int, List, str, int]], eval_func: Callable[..., float], *, n_trials: int = 50,
                opt_alpha: bool = False, lambda_range: Tuple[float, float] = (0.001, 1000.0),
                mu_range: Tuple[float, float] = (0.001, 1000.0), init_params: Optional[Dict[str, float]] = None,
                seed: Optional[int] = 42, ) -> Tuple[float, float, Optional[float]]:
    """在单个折上使用贝叶斯优化搜索 ``lambda``、``mu`` (及可选 ``alpha``).

    Parameters
    ----------
    seed : int, optional
        随机种子, 默认为 ``42`` 以保证结果可复现。当传入 ``None`` 或其他值时,
        将使用该种子初始化 :class:`~optuna.samplers.TPESampler`.
    """

    # --------- 初始化贝叶斯优化流程 --------- #
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    # 若提供了初始点, 则将其加入
    if init_params is not None:
        if opt_alpha and "alpha" not in init_params:
            init_params = {**init_params, "alpha": 1.0}
        study.enqueue_trial(init_params)
    pbar = tqdm(total=n_trials, desc="搜索进度", ncols=PROGRESS_NCOLS)

    # 回调函数: 每次 trial 完成后更新进度条显示当前结果
    def _callback(study: optuna.Study, trial: optuna.Trial) -> None:  # pragma: no cover
        pbar.update(1)
        postfix = {
            "lambda": f"{trial.params.get('lambda', float('nan')):.4f}",
            "mu": f"{trial.params.get('mu', float('nan')):.4f}",
            "acc": f"{trial.value:.4f}",
        }
        if opt_alpha:
            postfix["alpha"] = f"{trial.params.get('alpha', float('nan')):.4f}"
        pbar.set_postfix(postfix)

    # 运行贝叶斯优化迭代
    study.optimize(
        lambda trial: _objective(
            trial,
            samples,
            eval_func,
            opt_alpha=opt_alpha,
            lambda_range=lambda_range,
            mu_range=mu_range,
        ),
        n_trials=n_trials,
        callbacks=[_callback],
    )
    pbar.close()

    # 如果启用了 ``alpha`` 优化选项, 则直接返回最优结果
    if opt_alpha:
        lam_best = study.best_params["lambda"]
        mu_best = study.best_params["mu"]
        alpha_best = study.best_params.get("alpha")
        # 打印选出的最优超参组合
        tqdm.write(f"\nSelected best hyperparameters: lambda={lam_best:.4f}, mu={mu_best:.4f}, alpha={alpha_best:.4f}")
        return lam_best, mu_best, alpha_best

    # 未启用 ``alpha`` 优化时, 直接返回 Optuna 选出的最优组合
    lam_best = study.best_params["lambda"]
    mu_best = study.best_params["mu"]
    tqdm.write(f"\nSelected best hyperparameters: lambda={lam_best:.4f}, mu={mu_best:.4f}")
    return lam_best, mu_best, None


def tune_params(dataset: str, *, csv_path: str | Path | None = None, debug: bool = False,
                fold: Optional[int] = None, n_trials: int = 50, kfold: bool = False,
                opt_alpha: bool = False, twenty_times: bool = False, ) -> List[Dict[str, float]]:
    """搜索并保存最优的 ``lambda``、``mu`` (及可选 ``alpha``) 组合.

    Parameters
    ----------
    dataset : str
        数据集名称.
    kfold : bool, optional
        ``True`` 时按折号在验证集 ``bbavalset_<dataset>_fold<k>.csv`` 上逐折搜索,
        否则在完整数据集上评估 (默认).
    twenty_times : bool, optional
        若与 ``kfold`` 一同使用, 将整个 k-fold 搜索重复 20 次, 最终
        得到 100 组 ``(lambda, mu)`` 结果并写入 ``bayes_best_params_kfold_20times_<dataset>.csv``.
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
        lam_best, mu_best, alpha_best = search_fold(samples, eval_func, n_trials=n_trials, opt_alpha=opt_alpha)
        # 5. 将找到的最佳参数存入字典，并将结果四舍五入到小数点后 4 位
        result = {"lambda": round(lam_best, 4), "mu": round(mu_best, 4)}
        if opt_alpha and alpha_best is not None:
            result["alpha"] = round(alpha_best, 4)
        # 6. 构建结果文件的输出路径
        out_path = (
                Path(__file__).resolve().parents[1]
                / "experiments_result"
                / f"bayes_best_params_full_{dataset}.csv"
        )
        pd.DataFrame([result]).to_csv(out_path, index=False, encoding="utf-8", float_format="%.4f")
        if opt_alpha and alpha_best is not None:
            print(f"Full: lambda={lam_best:.4f}, mu={mu_best:.4f}, alpha={alpha_best:.4f}")
        else:
            print(f"Full: lambda={lam_best:.4f}, mu={mu_best:.4f}")
        return [result]

    # --------- 按 kfold 分折搜索 --------- #

    # 1. 加载交叉测试集, 仅用于确定可用的折号
    samples_cv = load_application_dataset_cv(csv_path=csv_path, debug=False)

    # 所有存在的折号, 供后续遍历
    folds = sorted(set(f for *_, f in samples_cv))
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

    # 若已存在完整数据集的最优解, 以其为中心缩小搜索范围并作为先验点
    lam0 = mu0 = None
    init_params: Optional[Dict[str, float]] = None
    lambda_range = (0.001, 1000.0)
    mu_range = (0.001, 1000.0)
    # 全局最优解先验 CSV 所在的地方
    full_csv = (Path(__file__).resolve().parents[1] / "experiments_result" / f"bayes_best_params_full_{dataset}.csv")
    if full_csv.exists():
        df_full = pd.read_csv(full_csv)
        lam0 = float(df_full.loc[0, "lambda"])
        mu0 = float(df_full.loc[0, "mu"])
        # 以全局最优解为中心, 将搜索范围缩小到 [lam0/1000, lam0*1000] 和 [mu0/1000, mu0*1000]
        lambda_range = (max(0.001, lam0 / 1000), min(1000.0, lam0 * 1000))
        mu_range = (max(0.001, mu0 / 1000), min(1000.0, mu0 * 1000))
        init_params = {"lambda": lam0, "mu": mu0}

    # 验证集所在目录
    val_dir = (Path(__file__).resolve().parents[1] / "data" / "bba_validation" / f"kfold_xu_bbavalset_{dataset}")

    if twenty_times:
        # 重复执行 20 次, 共生成 100 组 (lambda, mu)
        out_path = (Path(__file__).resolve().parents[
                        1] / "experiments_result" / f"bayes_best_params_kfold_20times_{dataset}.csv")
        results: List[Dict[str, float]] = []
        idx = 1
        for t in range(20):
            for f in folds:
                val_csv = val_dir / f"bbavalset_{dataset}_fold{f}.csv"
                val_samples_simple = load_application_dataset(csv_path=val_csv, debug=False)
                fold_samples = [(i, b, g, f) for i, b, g in val_samples_simple]
                lam_best, mu_best, _ = search_fold(
                    fold_samples,
                    eval_func,
                    n_trials=n_trials,
                    opt_alpha=opt_alpha,
                    lambda_range=lambda_range,
                    mu_range=mu_range,
                    init_params=init_params,
                    seed=42 + idx,
                )
                results.append({"times": idx, "lambda": round(lam_best, 4), "mu": round(mu_best, 4)})
                pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8", float_format="%.4f")
                idx += 1
        return results

    # 初始化一个空列表，用于存储每一折的最优参数结果
    results: List[Dict[str, float]] = []

    # 构建超参数保存路径
    out_path = (
            Path(__file__).resolve().parents[1]
            / "experiments_result"
            / f"bayes_best_params_kfold_{dataset}.csv"
    )

    # 遍历需要处理的每一折 (folds 列表可能是全部折，也可能是用户指定的单折)
    for f in folds:
        # 2. 加载当前折对应的验证集
        val_csv = val_dir / f"bbavalset_{dataset}_fold{f}.csv"
        val_samples_simple = load_application_dataset(csv_path=val_csv, debug=False)
        # 3. 为复用 search_fold, 添加折号字段
        fold_samples = [(i, b, g, f) for i, b, g in val_samples_simple]
        # 调用 search_fold 函数，在当前折的验证集上执行贝叶斯优化
        lam_best, mu_best, alpha_best = search_fold(
            fold_samples,
            eval_func,
            n_trials=n_trials,
            opt_alpha=opt_alpha,
            lambda_range=lambda_range,
            mu_range=mu_range,
            init_params=init_params,
        )
        # 将当前折的结果（折号、最佳lambda、最佳mu (及 alpha)）追加到 results 列表中，数值四舍五入到小数点后4位
        res = {"fold": f, "lambda": round(lam_best, 4), "mu": round(mu_best, 4)}
        if opt_alpha and alpha_best is not None:
            res["alpha"] = round(alpha_best, 4)
        results.append(res)
        # 在每完成一折的搜索后，都将当前的全部结果实时写入到 CSV 文件中，并打印
        pd.DataFrame(results).to_csv(out_path, index=False, encoding="utf-8", float_format="%.4f")
        if opt_alpha and alpha_best is not None:
            print(f"Fold {f}: lambda={lam_best:.4f}, mu={mu_best:.4f}, alpha={alpha_best:.4f}")
        else:
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
    # todo 是否同时优化 alpha 超参数
    parser.add_argument("--alpha", action="store_true", help="同时优化 alpha 超参", )
    # 指定是否运行在调试模式下
    parser.add_argument("--debug", action="store_true", help="调试模式, 仅运行少量迭代", )
    # 是否在 kfold 模式下重复运行 20 次
    parser.add_argument("--20times", dest="times20", action="store_true",
                        help="在 k-fold 模式下重复运行 20 次, 共生成 100 组 (lambda, mu)")
    args = parser.parse_args()

    # 进行调参
    tune_params(args.dataset, debug=args.debug, fold=args.fold, n_trials=args.trials, kfold=args.kfold,
                opt_alpha=args.alpha, twenty_times=args.times20, )
