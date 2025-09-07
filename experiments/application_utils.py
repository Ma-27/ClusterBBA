"""application_xxx 脚本共享的辅助函数。

本模块集中实现四个 ``application_*.py`` 脚本中复用的评估逻辑，包括收集预测结果、打印评估指标、按折使用不同超参进行评估以及超参数搜索脚本需要的 ``evaluate_accuracy`` 通用函数。

所有函数都接受 ``label_map`` 参数，以便调用者传入数据集的缩写标签与对应的完整标签。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize
from tabulate import tabulate
from tqdm import tqdm

import cluster.multi_clusters as multi_clusters
import config
import fusion.my_rule as my_rule
from utility.bba import BBA
from utility.io_application import load_application_dataset
from utility.probability import argmax, pignistic

__all__ = [
    "collect_predictions",
    "run_classification",
    "evaluate_accuracy",
    "load_kfold_params",
    "kfold_evaluate",
    "print_evaluation_matrix",
]


# ---------------------------------------------------------------------------
# 预测相关辅助函数
# ---------------------------------------------------------------------------


def collect_predictions(samples: List[Tuple[int, List[BBA], str]], combine_func: Callable[[List[BBA]], BBA],
                        label_map: Dict[str, str], *, show_progress: bool = True, warn: bool = False, ) -> Tuple[
    List[str], List[str], np.ndarray]:
    """根据给定融合函数生成真实标签、预测标签及概率矩阵。

    参数
    ----------
    samples :
        ``(索引, [BBA, ...], 真实标签)`` 形式的可迭代对象。
    combine_func :
        对单个样本的多条 BBA 进行融合的函数。
    label_map :
        缩写标签到完整标签的映射，便于阅读。
    show_progress :
        是否使用 tqdm 显示评估进度。
    warn :
        若融合失败，是否输出警告信息。
    """

    y_true: List[str] = []
    y_pred: List[str] = []
    y_score: List[List[float]] = []
    short_labels = list(label_map.keys())
    full_labels = [label_map[s] for s in short_labels]

    iterable: Iterable[Tuple[int, List[BBA], str]] = samples
    # tqdm 用于显示当前评估进度，列宽由全局配置 PROGRESS_NCOLS 控制
    if show_progress:
        iterable = tqdm(samples, desc="评估进度", ncols=config.PROGRESS_NCOLS)

    for idx, bbas, gt in iterable:
        # ---------- 预测流程 ---------- #
        # 1. 多条 BBA 先经指定规则融合
        try:
            fused = combine_func(bbas)
        except ValueError as e:  # pragma: no cover - 组合失败的异常分支
            if warn:
                print(f"样本 {idx} DS 组合失败: {e}")
            # 融合失败时记录一个错误标签，保证评估指标统计到该样本
            wrong_label = next(l for l in label_map.values() if l != gt)
            y_true.append(gt)
            y_pred.append(wrong_label)
            score = [0.0] * len(full_labels)
            score[full_labels.index(wrong_label)] = 1.0
            y_score.append(score)
            continue

        # 2. 对融合后的 BBA 进行 Pignistic 转换得到概率分布
        prob = pignistic(fused)
        # 3. 取概率最大的焦元作为预测类别
        fs, _ = argmax(prob)
        pred_short = next(iter(fs)) if fs else ""
        # 转换为完整标签，防止缩写难以阅读
        pred_full = label_map.get(pred_short, pred_short)

        # 记录真实标签与预测标签，供后续计算评估指标
        y_true.append(gt)
        y_pred.append(pred_full)
        y_score.append([prob.get_prob({s}) for s in short_labels])

    return y_true, y_pred, np.array(y_score)


def _print_metrics(y_true: List[str], y_pred: List[str], label_map: Dict[str, str], method_name: str, ) -> None:
    """打印准确率、F1 分数和混淆矩阵。"""

    labels = list(label_map.values())

    # 计算整体准确率与宏观 F1 分数
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    # 使用完整标签顺序计算各类别 F1，方便与表头一一对应
    f1_each = f1_score(y_true, y_pred, labels=labels, average=None)

    # 组装行数据并用 tabulate 生成 Markdown 表格，便于与文档直接结合
    rows = [[label, f"{score:.4f}"] for label, score in zip(labels, f1_each)]
    rows.append(["Accuracy", f"{acc:.4f}"])
    rows.append(["F1score", f"{f1_macro:.4f}"])
    df_res = pd.DataFrame(rows, columns=["Class / Metric", method_name])
    print(tabulate(df_res, headers="keys", tablefmt="pipe", showindex=False))

    # ---------------------- 混淆矩阵的计算与展示 ---------------------- #
    # cm[i, j] 表示真实类别 i 被预测成类别 j 的次数，可直观看出分类错配情况
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\nConfusion Matrix:")
    # 行列均为目标类别，直观展示预测错配情况
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())


def print_evaluation_matrix(y_true: List[str], y_pred: List[str], method_name: str, y_score: np.ndarray | None = None,
                            label_map: Dict[str, str] | None = None) -> None:
    """根据预测结果打印 TP、TN 等评估指标矩阵。"""

    # --------------- 基础计数（用 multilabel_confusion_matrix 构造混淆矩阵） --------------- #
    labels = list(label_map.values()) if label_map is not None else sorted(set(y_true) | set(y_pred))
    mcm = multilabel_confusion_matrix(y_true, y_pred, labels=labels)

    tp = mcm[:, 1, 1].sum()  # sum_i TP_i  #
    fp = mcm[:, 0, 1].sum()  # sum_i FP_i  # 预测成 i 类但真实非 i 类
    fn = mcm[:, 1, 0].sum()  # sum_i FN_i  # 真实为 i 类但预测非 i 类
    tn = mcm[:, 0, 0].sum()  # sum_i TN_i   ← 仍会>样本数，属于多分类定义

    # -------------------------- 派生指标 -------------------------- #
    # todo 使用宏平均（macro）以便与分类报告一致
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    # MCC 综合考虑 TP/TN/FP/FN，是二分类常用指标，这里直接调用库函数
    mcc = matthews_corrcoef(y_true, y_pred)

    # AUC：只有在提供 y_score（每个样本的类别概率或置信度向量）时才计算
    auc = np.nan
    if y_score is not None:
        y_true_bin = label_binarize(y_true, classes=labels)
        auc = roc_auc_score(y_true_bin, y_score, average="macro", multi_class="ovr" if len(labels) > 2 else "raise", )

    # --------------------------- 打印 --------------------------- #
    df = pd.DataFrame({
        "": ["Precision", "Recall", "MCC", "AUC"],
        method_name: [precision, recall, mcc, auc],
    })
    print(df.to_markdown(index=False, floatfmt=".4f"))


# ---------------------------------------------------------------------------
# 对外暴露的通用函数
# ---------------------------------------------------------------------------


def run_classification(samples: List[Tuple[int, List[BBA], str]], combine_func: Callable[[List[BBA]], BBA],
                       label_map: Dict[str, str], method_name: str, *, warn: bool = True, ) -> Tuple[
    List[str], List[str], np.ndarray]:
    """收集预测结果并输出评估指标, 返回 ``(y_true, y_pred, y_score)``。"""

    # 调用统一的预测函数获取标签
    y_true, y_pred, y_score = collect_predictions(
        samples, combine_func, label_map, show_progress=True, warn=warn
    )
    _print_metrics(y_true, y_pred, label_map, method_name)
    # 返回真实标签、预测标签与概率矩阵，供外部继续计算其他指标
    return y_true, y_pred, y_score


def evaluate_accuracy(*, samples: List[Tuple[int, List[BBA], str]] | None = None, debug: bool = False,
                      show_progress: bool = False, csv_path: str | Path | None = None,
                      combine_func: Callable[[List[BBA]], BBA] = my_rule.my_combine, data_progress: bool = True,
                      warn: bool = False, label_map: Dict[str, str], ) -> float:
    """供超参数搜索脚本复用的通用准确率计算函数。"""

    if samples is None:
        if csv_path is None:
            raise ValueError("samples 与 csv_path 至少需提供其一")
        samples = load_application_dataset(
            debug=debug, csv_path=csv_path, show_progress=data_progress
        )

    y_true, y_pred, _ = collect_predictions(
        samples, combine_func, label_map, show_progress=show_progress, warn=warn
    )
    return float(accuracy_score(y_true, y_pred))


def load_kfold_params(csv_path: str | Path) -> Dict[int, Tuple[float, float]]:
    """从 CSV 文件读取 ``fold -> (lambda, mu)`` 的映射关系。"""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"未找到超参数文件: {csv_path}")
    df = pd.read_csv(csv_path)
    return {
        int(row["fold"]): (float(row["lambda"]), float(row["mu"]))
        for _, row in df.iterrows()
    }


def _apply_hyperparams(lambda_val: float, mu_val: float) -> None:
    """将 λ 和 μ 更新到相关模块的全局变量。"""

    config.LAMBDA = lambda_val
    config.MU = mu_val
    multi_clusters.LAMBDA = lambda_val
    multi_clusters.MU = mu_val
    my_rule.LAMBDA = lambda_val
    my_rule.MU = mu_val


def kfold_evaluate(samples_cv: List[Tuple[int, List[BBA], str, int]], param_map: Dict[int, Tuple[float, float]],
                   label_map: Dict[str, str], combine_func: Callable[[List[BBA]], BBA], *,
                   method_name: str = "Proposed", warn: bool = True, ) -> Tuple[List[str], List[str]]:
    """在 Proposed 方法下按折使用最优超参进行评估并返回预测结果。"""

    # 收集所有折的真实标签、预测标签及概率
    y_true_all: List[str] = []
    y_pred_all: List[str] = []
    y_score_all: List[List[float]] = []
    # 遍历样本中出现的所有折号
    folds = sorted({fold for _, _, _, fold in samples_cv})
    for fold in folds:
        if fold not in param_map:
            raise KeyError(f"超参文件缺少第 {fold} 折的参数")
        # 获取该折的 (λ, μ) 超参
        lambda_val, mu_val = param_map[fold]
        # 将超参写入相关模块
        _apply_hyperparams(lambda_val, mu_val)
        # 提取当前折的样本
        fold_samples = [
            (i, b, gt) for i, b, gt, f in samples_cv if f == fold
        ]
        # 收集当前折的预测结果
        y_t, y_p, y_s = collect_predictions(
            fold_samples, combine_func, label_map, show_progress=True, warn=warn
        )
        # 累加所有折的预测结果
        y_true_all.extend(y_t)
        y_pred_all.extend(y_p)
        y_score_all.extend(y_s.tolist())

    # 输出整体评估指标
    _print_metrics(y_true_all, y_pred_all, label_map, method_name)
    return y_true_all, y_pred_all, np.array(y_score_all)
