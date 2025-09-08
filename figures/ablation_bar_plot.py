# -*- coding: utf-8 -*-
"""消融实验分组柱状图
====================
读取 ``experiments_result/ablation_results.csv``，根据参数绘制 Accuracy 或 F1 Score 的分组柱状图。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 保证可以导入项目根目录下的模块
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 统一绘图风格
from utility.plot_style import apply_style
from utility.plot_utils import savefig

apply_style()

# 结果数据存放目录
RES_DIR = os.path.join(BASE_DIR, "experiments_result")
CSV_PATH = os.path.join(RES_DIR, "ablation_results.csv")

# 配置与数据集的顺序及显示标签
CONFIG_LABELS: List[tuple[str, str]] = [
    ("baseline", "Baseline"),
    ("mu_lambda", r"w/ $(\mu,\lambda)$"),
    ("alpha", r"w/ $\alpha$"),
    ("full", "Fully Opt."),
]
DATASET_ORDER: List[str] = ["Iris", "Wine", "Seeds", "Glass"]


def plot(metric: str = "accuracy") -> None:
    """绘制指定指标的分组柱状图

    Parameters
    ----------
    metric: str
        指标类型，可选 ``accuracy`` 或 ``f1``。
    """

    if metric not in {"accuracy", "f1"}:
        raise ValueError("metric must be 'accuracy' or 'f1'")

    df = pd.read_csv(CSV_PATH)
    # 生成以数据集为行、配置为列的透视表
    pivot = df.pivot(index="dataset", columns="config", values=metric)
    pivot = pivot.reindex(DATASET_ORDER)

    x = np.arange(len(DATASET_ORDER))
    group_width = 0.8  # 单组柱状图在 x 轴上的总宽度
    bar_width = group_width / len(CONFIG_LABELS)
    offsets = np.linspace(
        -group_width / 2 + bar_width / 2,
        group_width / 2 - bar_width / 2,
        len(CONFIG_LABELS),
    )

    fig, ax = plt.subplots(figsize=(12 / 2.54, 7 / 2.54))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, zorder=0, linestyle="--", linewidth=0.5)

    # 使用对比强烈且美观的配色
    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2"]
    for i, (config, label) in enumerate(CONFIG_LABELS):
        ax.bar(
            x + offsets[i],
            pivot[config].to_numpy(),
            width=bar_width,
            color=colors[i % len(colors)],
            label=label,
            zorder=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(DATASET_ORDER)
    ax.set_ylabel("F1 Score" if metric == "f1" else "Accuracy")
    ax.set_ylim(0.0, 1.0)

    title_metric = "F1 Score" if metric == "f1" else "Accuracy"
    fig.suptitle(
        f"Ablation Experiment Results on Overall {title_metric} across Four Datasets.",
        y=0.98,
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(CONFIG_LABELS),
        frameon=True,
        bbox_to_anchor=(0.5, 0.9),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.82])

    out_path = os.path.join(RES_DIR, f"ablation_{metric}.png")
    savefig(fig, out_path, tight_layout=False)


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="绘制消融实验分组柱状图")
    parser.add_argument(
        "--metric",
        choices=["accuracy", "f1"],
        default="accuracy",
        help="选择绘制的性能指标",
    )
    args = parser.parse_args()
    plot(args.metric)
