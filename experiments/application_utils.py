"""application_xxx 脚本共享的辅助函数。

本模块集中实现四个 ``application_*.py`` 脚本中复用的评估逻辑，包括收集预测结果、打印评估指标、按折使用不同超参进行评估以及超参数搜索脚本需要的 ``evaluate_accuracy`` 通用函数。

所有函数都接受 ``label_map`` 参数，以便调用者传入数据集的缩写标签与对应的完整标签。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
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
]


# ---------------------------------------------------------------------------
# 预测相关辅助函数
# ---------------------------------------------------------------------------


def collect_predictions(samples: List[Tuple[int, List[BBA], str]], combine_func: Callable[[List[BBA]], BBA],
                        label_map: Dict[str, str], *, show_progress: bool = True, warn: bool = False, ) -> Tuple[
    List[str], List[str]]:
    """根据给定融合函数生成真实标签与预测标签列表。

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

    return y_true, y_pred


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

    # 混淆矩阵：cm[i, j] 表示真实类别 i 被预测成类别 j 的次数
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\nConfusion Matrix:")
    # 行列均为目标类别，直观展示预测错配情况
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())


# ---------------------------------------------------------------------------
# 对外暴露的通用函数
# ---------------------------------------------------------------------------


def run_classification(samples: List[Tuple[int, List[BBA], str]], combine_func: Callable[[List[BBA]], BBA],
                       label_map: Dict[str, str], method_name: str, *, warn: bool = True, ) -> None:
    """收集预测结果并输出评估指标。"""

    # 调用统一的预测函数获取标签
    y_true, y_pred = collect_predictions(
        samples, combine_func, label_map, show_progress=True, warn=warn
    )
    _print_metrics(y_true, y_pred, label_map, method_name)


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

    y_true, y_pred = collect_predictions(
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
                   method_name: str = "Proposed", warn: bool = True, ) -> None:
    """在 Proposed 方法下按折使用最优超参进行评估。"""

    # 收集所有折的真实标签与预测标签
    y_true_all: List[str] = []
    y_pred_all: List[str] = []
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
        y_t, y_p = collect_predictions(
            fold_samples, combine_func, label_map, show_progress=True, warn=warn
        )
        # 累加所有折的预测结果
        y_true_all.extend(y_t)
        y_pred_all.extend(y_p)

    # 输出整体评估指标
    _print_metrics(y_true_all, y_pred_all, label_map, method_name)
