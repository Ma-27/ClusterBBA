# -*- coding: utf-8 -*-
"""Iris 数据集上的证据融合分类实验
=================================

使用 :func:`utility.io_application.load_application_dataset` 读取``kfold_xu_bba_iris.csv``，对每个样本的多条 BBA 按指定融合规则组合，再经Pignistic 转换得到预测类别，最终计算准确率和 F1 分数。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, List

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tabulate import tabulate
from tqdm import tqdm

# 确保包导入路径指向项目根目录
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 依赖本项目内现成工具函数 / 模块
from config import PROGRESS_NCOLS
from fusion.ds_rule import combine_multiple
from fusion.murphy_rule import murphy_combine
from fusion.deng_mae_rule import modified_average_evidence
from fusion.xiao_bjs_rule import xiao_bjs_combine
from fusion.xiao_rb_rule import xiao_rb_combine
from fusion.my_rule import my_combine
from utility.io_application import load_application_dataset
from utility.probability import pignistic, argmax
from utility.bba import BBA

LABEL_MAP = {"Se": "Setosa", "Ve": "Versicolor", "Vi": "Virginica"}

# 可选融合规则及其对应的实现函数
METHODS = {
    "Dempster": combine_multiple,
    "Murphy": murphy_combine,
    "Deng": modified_average_evidence,
    "Xiao BJS": xiao_bjs_combine,
    "Xiao RB": xiao_rb_combine,
    "Proposed": my_combine,
}


def run_classification(samples: List[tuple[int, List, str]],
                       combine_func: Callable[[List[BBA]], BBA],
                       method_name: str) -> None:
    y_true: List[str] = []
    y_pred: List[str] = []

    # tqdm 用于显示当前评估进度，PROGRESS_NCOLS 来源于全局配置
    pbar = tqdm(samples, desc="评估进度", ncols=PROGRESS_NCOLS)
    for _, bbas, gt in pbar:
        # ---------- 预测流程 ---------- #
        # 1. 多条 BBA 先经指定规则融合
        fused = combine_func(bbas)
        # 2. 对融合后的 BBA 进行 Pignistic 转换得到概率分布
        prob = pignistic(fused)
        # 3. 取概率最大的焦元作为预测结果
        fs, _ = argmax(prob)
        pred_short = next(iter(fs)) if fs else ""
        # 转换为完整标签，防止缩写难以阅读
        pred_full = LABEL_MAP.get(pred_short, pred_short)

        # 记录真实标签与预测标签，供后续计算评估指标
        y_true.append(gt)
        y_pred.append(pred_full)

    # ------------------------------ 评估指标 ------------------------------ #
    # 计算整体准确率与宏观 F1 分数
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")

    # 使用完整标签顺序计算各类别 F1，方便与表头一一对应
    labels = list(LABEL_MAP.values())
    f1_each = f1_score(y_true, y_pred, labels=labels, average=None)

    # 组装行数据并用 tabulate 生成 Markdown 表格，便于与文档直接结合
    rows = [[label, f"{score:.4f}"] for label, score in zip(labels, f1_each)]
    rows.append(["Accuracy", f"{acc:.4f}"])
    rows.append(["F1score", f"{f1_macro:.4f}"])
    df_res = pd.DataFrame(rows, columns=["Class / Metric", method_name])
    print(tabulate(df_res, headers="keys", tablefmt="pipe", showindex=False))

    # ------------------------------ 混淆矩阵 ------------------------------ #
    # cm[i, j] 表示真实类别 i 被预测成类别 j 的次数
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\nConfusion Matrix:")
    # 行列均为目标类别，直观展示预测错配情况
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="在 Iris 数据集上进行证据融合分类")
    parser.add_argument("--full", action="store_true",
                        help="评估全部 150 条样本")

    # fixme 指定融合规则：从 METHODS 字典中取出对应的函数和名称
    parser.add_argument(
        "--method",
        type=str,
        choices=list(METHODS.keys()),
        default="Proposed",
        help="选择融合规则，此处可以任意更改",
    )
    args = parser.parse_args()

    # fixme 如果 --full 参数未指定，则仅评估前 2 条样本
    # debug = not args.full
    debug = args.full

    # fixme CSV 文件路径，根据实验灵活修改
    csv_path = Path(__file__).resolve().parents[1] / "data" / "kfold_xu_bba_iris.csv"

    method_name = args.method
    combine_func = METHODS[method_name]

    # 载入数据集
    samples = load_application_dataset(debug=debug, csv_path=csv_path)
    # 进行分类任务评估
    combine_func = METHODS[args.method]
    run_classification(samples, combine_func, args.method)
