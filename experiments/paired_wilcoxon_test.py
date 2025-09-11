# -*- coding: utf-8 -*-
"""不同方法在同一数据集上的配对 Wilcoxon 符号秩检验
=============================================

本脚本使用 5 折交叉验证得到的准确率，对任意两种证据融合方法进行配对样本Wilcoxon 符号秩检验，输出统计量和对应的 p 值。数据集、待比较的方法以及是否启用按折超参数均可通过命令行参数指定。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from scipy.stats import wilcoxon
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# 确保包导入路径指向项目根目录
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 依赖本项目内的工具函数与数据集定义
from experiments import (
    application_glass,
    application_iris,
    application_seeds,
    application_wine,
)
from experiments.application_utils import (
    collect_predictions,
    load_kfold_params,
    _apply_hyperparams,
)
from utility.io_application import load_application_dataset_cv

# ---------------------------------------------------------------------------
# 数据集与默认基线配置
# ---------------------------------------------------------------------------

DATASET_CONFIG: Dict[str, Dict[str, object]] = {
    "iris": {
        "module": application_iris,
        "csv": Path(__file__).resolve().parents[1] / "data" / "kfold_xu_bba_iris.csv",
        "params": Path(__file__).resolve().parents[1]
                  / "experiments_result"
                  / "bayes_best_params_kfold_iris.csv",
        "baseline": "Deng",
    },
    # 注意这个超参数选用的是网格搜索出来的超参数对
    "wine": {
        "module": application_wine,
        "csv": Path(__file__).resolve().parents[1] / "data" / "kfold_xu_bba_wine.csv",
        "params": Path(__file__).resolve().parents[1]
                  / "experiments_result"
                  / "best_params_kfold_wine.csv",
        "baseline": "Dempster",
    },
    "seeds": {
        "module": application_seeds,
        "csv": Path(__file__).resolve().parents[1] / "data" / "kfold_xu_bba_seeds.csv",
        "params": Path(__file__).resolve().parents[1]
                  / "experiments_result"
                  / "bayes_best_params_kfold_seeds.csv",
        "baseline": "Dempster",
    },
    "glass": {
        "module": application_glass,
        "csv": Path(__file__).resolve().parents[1] / "data" / "kfold_xu_bba_glass_resplit.csv",
        "params": Path(__file__).resolve().parents[1]
                  / "experiments_result"
                  / "bayes_best_params_kfold_glass.csv",
        "baseline": "Xiao BJS",
    },
}


# ---------------------------------------------------------------------------
# 评估与统计检验核心逻辑
# ---------------------------------------------------------------------------


def _evaluate_method(samples: List[Tuple[int, List, str, int]], method_name: str, *, label_map: Dict[str, str],
                     methods: Dict[str, object], use_kfold_params: bool, params_file: Path,
                     lambda_val: float | None, mu_val: float | None, alpha_val: float | None) -> List[float]:
    """计算给定方法在各折上的准确率列表."""

    combine_func = methods[method_name]
    folds = sorted({fold for _, _, _, fold in samples})

    # 若启用按折超参数, 则预先加载所有折的 (lambda, mu)
    param_map = {}
    if method_name == "Proposed":
        if use_kfold_params:
            if not params_file.is_file():
                raise FileNotFoundError(f"缺少超参数文件: {params_file}")
            param_map = load_kfold_params(params_file)
        else:
            if None in (lambda_val, mu_val, alpha_val):
                raise ValueError("未启用 kfold 时必须提供 lambda、mu、alpha")
            _apply_hyperparams(lambda_val, mu_val, alpha_val)

    accs: List[float] = []
    # 遍历样本中出现的所有折号
    for f in folds:
        fold_samples = [(i, b, gt) for i, b, gt, fold in samples if fold == f]
        # 若需要, 根据当前折号加载对应的 (lambda, mu)
        if method_name == "Proposed" and use_kfold_params:
            # 确保超参数文件中包含该折的参数
            if f not in param_map:
                raise KeyError(f"超参数文件缺少第 {f} 折的参数")
            # 获取该折的 (λ, μ) 超参
            l_val, m_val = param_map[f]
            if alpha_val is None:
                raise ValueError("启用 kfold 时必须指定 alpha")
            _apply_hyperparams(l_val, m_val, alpha_val)
        y_true, y_pred, _ = collect_predictions(
            fold_samples, combine_func, label_map, show_progress=False, warn=False
        )
        accs.append(float(accuracy_score(y_true, y_pred)))
    return accs


def paired_wilcoxon_test(dataset: str = "iris", method1: str = "Proposed", method2: str = "Proposed", *,
                         use_kfold1: bool = False, use_kfold2: bool = False,
                         lambda1: float | None = None, mu1: float | None = None, alpha1: float | None = None,
                         lambda2: float | None = None, mu2: float | None = None, alpha2: float | None = None,
                         two_sided: bool = False,
                         ) -> Tuple[float, float]:
    """执行配对 Wilcoxon 符号秩检验并返回 ``(statistic, p_value)``。默认在单侧检验下判断 ``method2`` 是否显著优于 ``method1``。"""

    if dataset not in DATASET_CONFIG:
        raise ValueError(f"未知数据集: {dataset}")
    # 获取数据集配置
    cfg = DATASET_CONFIG[dataset]
    if method2 is None:
        method2 = cfg["baseline"]  # 默认与该数据集表现最佳的基线比较

    module = cfg["module"]
    csv_path: Path = cfg["csv"]  # type: ignore[assignment]
    params_file: Path = cfg["params"]  # type: ignore[assignment]

    # 载入包含折号的数据集
    samples_cv = load_application_dataset_cv(csv_path=csv_path, debug=False)

    # 分别计算两种方法在每个折上的准确率
    accs1 = _evaluate_method(samples_cv, method1, label_map=module.LABEL_MAP, methods=module.METHODS,
                             use_kfold_params=use_kfold1, params_file=params_file,
                             lambda_val=lambda1, mu_val=mu1, alpha_val=alpha1)
    accs2 = _evaluate_method(samples_cv, method2, label_map=module.LABEL_MAP, methods=module.METHODS,
                             use_kfold_params=use_kfold2, params_file=params_file,
                             lambda_val=lambda2, mu_val=mu2, alpha_val=alpha2)

    # 使用 scipy 进行配对样本 Wilcoxon 符号秩检验，交换输入顺序以便正的统计量意味着 ``method2`` 更优
    alt = "two-sided" if two_sided else "greater"
    stat, p_val = wilcoxon(accs2, accs1, alternative=alt)
    return float(stat), float(p_val)


# ---------------------------------------------------------------------------
# 命行入口
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="配对 Wilcoxon 符号秩检验")
    # --dataset：指定需要进行检验的数据集名称
    # todo 可选值包括 iris、wine、seeds、glass，默认为 iris。
    parser.add_argument("--dataset", type=str, choices=list(DATASET_CONFIG.keys()), default="iris",
                        help="选择进行检验的数据集", )

    # --method1：第一个待比较的方法名称，默认为 Proposed
    parser.add_argument("--method1", type=str, default="Proposed", help="第一个待比较的方法名称", )

    # --method2：第二个待比较的方法名称，默认为 Proposed
    # 若需与基线方法比较，可指定基线名称
    parser.add_argument("--method2", type=str, default="Proposed", help="第二个待比较的方法名称", )

    # --kfold1：提供该标志时，方法1为每折加载对应的超参数
    parser.add_argument("--kfold1", action="store_true", help="方法1是否启用按折超参数", )

    # --kfold2：提供该标志时，方法2为每折加载对应的超参数
    parser.add_argument("--kfold2", action="store_true", help="方法2是否启用按折超参数", )

    # --lambda1 / --mu1 / --alpha1：方法1的超参数
    # 若未启用 --kfold1，这三项必须全部指定；
    # 启用 --kfold1 时只需提供 --alpha1 作为全局 α
    parser.add_argument("--lambda1", type=float, default=None, help="方法1的 λ", )
    parser.add_argument("--mu1", type=float, default=None, help="方法1的 μ", )
    parser.add_argument("--alpha1", type=float, default=None, help="方法1的 α", )

    # --lambda2 / --mu2 / --alpha2：方法2的超参数，规则与方法1一致
    parser.add_argument("--lambda2", type=float, default=None, help="方法2的 λ", )
    parser.add_argument("--mu2", type=float, default=None, help="方法2的 μ", )
    parser.add_argument("--alpha2", type=float, default=None, help="方法2的 α", )

    # --two-sided：若提供该标志，则执行双侧检验；默认执行单侧检验
    parser.add_argument("--two-sided", action="store_true", help="使用双侧检验 (默认单侧)", )

    args = parser.parse_args()

    if args.kfold1:
        if args.alpha1 is None:
            parser.error("启用 --kfold1 时必须提供 --alpha1")
    else:
        for p in ("lambda1", "mu1", "alpha1"):
            if getattr(args, p) is None:
                parser.error("未启用 --kfold1 时需提供 --lambda1、--mu1、--alpha1")

    if args.kfold2:
        if args.alpha2 is None:
            parser.error("启用 --kfold2 时必须提供 --alpha2")
    else:
        for p in ("lambda2", "mu2", "alpha2"):
            if getattr(args, p) is None:
                parser.error("未启用 --kfold2 时需提供 --lambda2、--mu2、--alpha2")

    if args.method1 == args.method2 == "Proposed":
        if args.kfold1 and args.kfold2:
            if args.alpha1 == args.alpha2:
                parser.error("Proposed 方法的两组超参数完全一致，无法比较")
        elif not args.kfold1 and not args.kfold2:
            if args.lambda1 == args.lambda2 and args.mu1 == args.mu2 and args.alpha1 == args.alpha2:
                parser.error("Proposed 方法的两组超参数完全一致，无法比较")

    # 执行配对 Wilcoxon 符号秩检验
    stat, p_val = paired_wilcoxon_test(dataset=args.dataset, method1=args.method1, method2=args.method2,
                                       use_kfold1=args.kfold1, use_kfold2=args.kfold2,
                                       lambda1=args.lambda1, mu1=args.mu1, alpha1=args.alpha1,
                                       lambda2=args.lambda2, mu2=args.mu2, alpha2=args.alpha2,
                                       two_sided=args.two_sided)
    # 输出结果
    tail = "two-sided" if args.two_sided else "one-sided"
    extra = "" if args.two_sided else f", testing {args.method2} > {args.method1}"
    print(
        f"Paired Wilcoxon signed-rank test ({tail}{extra}) on {args.dataset} between {args.method1} and {args.method2}:")
    print(f"statistic = {stat:.4f}, p-value = {p_val:.4f}")
