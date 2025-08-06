# -*- coding: utf-8 -*-
"""Glass 数据集上的证据融合分类实验
=================================

使用 :func:`utility.io_application.load_application_dataset` 读取``xu_bba_glass.csv``，对每个样本的多条 BBA 按指定融合规则组合，再经 Pignistic 转换得到预测类别，最终计算准确率和 F1 分数。
原来研究使用的 cmd 参数，Glass 数据集很特殊，永远不启用 kfold ：--method Proposed --kfold
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
    kfold_evaluate,
)

# ---------------------------------------------------------------------------
# 数据集标签与可选融合方法
# ---------------------------------------------------------------------------

LABEL_MAP = {
    "Bf": "building_windows_float_processed",
    "Bn": "building_windows_non_float_processed",
    "Vf": "vehicle_windows_float_processed",
    "Co": "containers",
    "Ta": "tableware",
    "He": "headlamps",
}

# 可选融合规则及其对应的实现函数
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
    """计算在 Glass 数据集上的分类准确率."""

    if samples is None and csv_path is None:
        csv_path = Path(__file__).resolve().parents[1] / "data" / "xu_bba_glass.csv"
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
    parser = argparse.ArgumentParser(
        description="在 Glass 数据集上进行证据融合分类"
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="评估全部 214 条样本",
    )

    # fixme 指定融合规则：从 METHODS 字典中取出对应的函数和名称
    parser.add_argument(
        "--method",
        type=str,
        choices=list(METHODS.keys()),
        default="Proposed",
        help="选择融合规则，此处可以任意更改",
    )

    # fixme 指定是否使用 K 折交叉验证的特殊超参数
    parser.add_argument(
        "--kfold",
        action="store_true",
        help="Proposed 方法按折使用最优超参评估",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="覆盖 config.py 中的 ALPHA 超参",
    )

    args = parser.parse_args()

    # 若指定 alpha, 则覆盖 config.py 中的默认值
    if args.alpha is not None:
        config.ALPHA = args.alpha
        try:
            import fusion.my_rule as my_rule

            my_rule.ALPHA = args.alpha
        except Exception:  # pragma: no cover - 导入失败时忽略
            pass

    # fixme 如果 --full 参数未指定，则仅评估前 2 条样本
    # debug = not args.full
    debug = args.full

    data_dir = Path(__file__).resolve().parents[1] / "data"
    csv_path = data_dir / ("kfold_xu_bba_glass.csv" if args.kfold else "xu_bba_glass.csv")

    if args.method == "Proposed" and args.kfold:
        # ------------------------------ K 折评估 ------------------------------ #
        params_path = (
                Path(__file__).resolve().parents[1]
                / "experiments_result"
                / "kfold_best_params_glass.csv"
        )
        if not params_path.exists():
            raise FileNotFoundError(f"缺少超参数文件: {params_path}")
        param_map = load_kfold_params(params_path)
        # 载入数据集
        samples_cv = load_application_dataset_cv(debug=debug, csv_path=csv_path)
        # 进行分类任务评估
        kfold_evaluate(samples_cv, param_map, LABEL_MAP, METHODS["Proposed"])
    else:
        # ---------------------------- 单次评估流程 ---------------------------- #
        combine_func = METHODS[args.method]
        samples = load_application_dataset(debug=debug, csv_path=csv_path)
        # 进行分类任务评估
        run_classification(samples, combine_func, LABEL_MAP, args.method)
