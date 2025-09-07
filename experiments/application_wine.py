# -*- coding: utf-8 -*-
"""Wine 数据集上的证据融合分类实验
=================================

使用 :func:`utility.io_application.load_application_dataset` 读取``kfold_xu_bba_wine.csv``，对每个样本的多条 BBA 按指定融合规则组合，再经 Pignistic 转换得到预测类别，最终计算准确率和 F1 分数。
原来研究使用的 cmd 参数：--method Proposed --kfold
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, List

# 确保包导入路径指向项目根目录
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 依赖本项目内现成工具函数 / 模块
import config
from fusion.ds_rule import combine_multiple
from fusion.murphy_rule import murphy_combine
from fusion.deng_mae_rule import modified_average_evidence
from fusion.xiao_bjs_rule import xiao_bjs_combine
from fusion.xiao_rb_rule import xiao_rb_combine
from fusion.xiao_bjs_pure_rule import xiao_bjs_pure_combine
from fusion.my_rule import my_combine
from utility.io_application import (
    load_application_dataset,
    load_application_dataset_cv,
)
from utility.bba import BBA
from experiments.application_utils import (
    run_classification,
    evaluate_accuracy as _evaluate_accuracy,
    load_kfold_params,
    kfold_evaluate, print_evaluation_matrix,
)

# ---------------------------------------------------------------------------
# 数据集标签与可选融合方法
# ---------------------------------------------------------------------------

LABEL_MAP = {"C1": "Class 1", "C2": "Class 2", "C3": "Class 3"}

METHODS = {
    "Dempster": combine_multiple,
    "Murphy": murphy_combine,
    "Deng": modified_average_evidence,
    "Xiao BJS": xiao_bjs_combine,
    "Xiao BJS Pure": xiao_bjs_pure_combine,
    "Xiao RB": xiao_rb_combine,
    "Proposed": my_combine,
}


# ---------------------------------------------------------------------------
# 对外暴露的评估函数
# ---------------------------------------------------------------------------


def evaluate_accuracy(*, samples: List[tuple[int, List[BBA], str]] | None = None, debug: bool = False,
                      show_progress: bool = False, csv_path: str | Path | None = None,
                      combine_func: Callable[[List[BBA]], BBA] = my_combine, data_progress: bool = True,
                      warn: bool = False, ) -> float:
    """计算在 Wine 数据集上的分类准确率."""

    if samples is None and csv_path is None:
        csv_path = (
                Path(__file__).resolve().parents[1] / "data" / "kfold_xu_bba_wine.csv"
        )
    return _evaluate_accuracy(
        samples=samples,
        debug=debug,
        show_progress=show_progress,
        csv_path=csv_path,
        combine_func=combine_func,
        data_progress=data_progress,
        warn=warn,
        label_map=LABEL_MAP,
    )


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在 Wine 数据集上进行证据融合分类")
    # 指定是否运行在调试模式下
    parser.add_argument("--debug", action="store_true", help="仅评估前 2 条样本", )
    # todo 评估不同的融合规则
    parser.add_argument("--method", type=str, choices=list(METHODS.keys()), default="Proposed",
                        help="选择融合规则，此处可以任意更改", )
    # todo 指定是否启用 K 折交叉验证模式
    parser.add_argument("--kfold", action="store_true", help="Proposed 方法按折使用最优超参评估", )
    # 使用特定的 alpha 超参数
    parser.add_argument("--alpha", type=float, default=None, help="覆盖 config.py 中的 ALPHA 超参", )
    args = parser.parse_args()

    # 若指定 alpha, 则覆盖 config.py 中的默认值
    if args.alpha is not None:
        config.ALPHA = args.alpha
        try:
            import fusion.my_rule as my_rule

            my_rule.ALPHA = args.alpha
        except Exception:  # pragma: no cover - 导入失败时忽略
            pass

    # 启用调试模式时仅评估前 2 条样本
    debug = args.debug

    csv_path = Path(__file__).resolve().parents[1] / "data" / "kfold_xu_bba_wine.csv"

    if args.method == "Proposed" and args.kfold:
        # ------------------------------ K 折评估 ------------------------------ #
        params_path = (
                Path(__file__).resolve().parents[1]
                / "experiments_result"
                # / "bayes_best_params_kfold_wine.csv"   # 修改这里以启用旧版本的最优化超参数
                / "best_params_kfold_wine.csv"
        )
        if not params_path.exists():
            raise FileNotFoundError(f"缺少超参数文件: {params_path}")
        param_map = load_kfold_params(params_path)
        # 载入数据集
        samples_cv = load_application_dataset_cv(debug=debug, csv_path=csv_path)
        # 进行分类任务评估，收集预测结果
        y_true, y_pred, y_score = kfold_evaluate(
            samples_cv, param_map, LABEL_MAP, METHODS["Proposed"]
        )
    else:
        # ---------------------------- 单次评估流程 ---------------------------- #
        combine_func = METHODS[args.method]
        # 载入数据集
        samples = load_application_dataset(debug=debug, csv_path=csv_path)
        # 进行分类任务评估，收集预测结果
        y_true, y_pred, y_score = run_classification(
            samples, combine_func, LABEL_MAP, args.method
        )

    # 在前述常规评估后，额外输出 TP、TN、Precision 等更细致的指标矩阵
    print("\nAdditional Evaluation Metrics:")
    print_evaluation_matrix(y_true, y_pred, args.method, y_score=y_score, label_map=LABEL_MAP)
