# -*- coding: utf-8 -*-
"""可视化贝叶斯调参结果
=================================
读取 ``bayes_best_params_kfold_20times_<dataset>.csv``，分别绘制``mu`` 与 ``lambda`` 两个参数在不同试验次数下的取值。纵轴采用``log10`` 变换并以 ``0`` 为对称轴，完整显示 ``[-3, 3]`` 范围。主图长宽比为 ``18:5``（不含图例），图例置于右侧框外。
"""

from __future__ import annotations

import os
import string
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 确保导入路径指向项目根目录
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 依赖本项目内现成工具函数 / 模块
from utility.plot_style import apply_style
from utility.plot_utils import savefig

# 统一绘图风格
apply_style()

# 实验结果存放目录
RES_DIR = os.path.join(BASE_DIR, "experiments_result")

# 数据集的元信息：标题等
DATASET_META: dict[str, dict[str, str]] = {
    "iris": {"title": "Iris"},
    "wine": {"title": "Wine"},
    "seeds": {"title": "Seeds"},
    "glass": {"title": "Glass"},
}

# 数据集与标题字母编号的映射关系
DATASET_LABEL_IDX: dict[str, int] = {
    "iris": 0,  # (a)
    "wine": 1,  # (b)
    "seeds": 2,  # (c)
    "glass": 3,  # (d)
}


def plot_dataset(dataset: str, label_idx: int, font_kwargs: dict | None = None) -> None:
    """绘制单个数据集的 ``mu`` 与 ``lambda`` 曲线

    Parameters
    ----------
    dataset:
        数据集名称，对应 csv 文件前缀。
    label_idx:
        字母编号，用于在标题中标记 ``(a)``, ``(b)`` 等。
    font_kwargs:
        控制标题和坐标轴字体的大小与粗细，例如 ``{"fontsize": 14, "fontweight": "bold"}``。
    """

    # 构造 csv 路径并检查文件是否存在
    csv_path = os.path.join(RES_DIR, f"bayes_best_params_kfold_20times_{dataset}.csv")
    if not os.path.isfile(csv_path):
        print(f"CSV 文件不存在：{csv_path}")
        return

    # 读取 csv，第一列为试验次数，其余列为待绘制参数
    df = pd.read_csv(csv_path)
    if not {"lambda", "mu"}.issubset(df.columns):
        print(f"CSV 缺少必要的列：{csv_path}")
        return

    x = df.iloc[:, 0].to_numpy()
    y_lambda = np.log10(df["lambda"].to_numpy())
    y_mu = np.log10(df["mu"].to_numpy())

    # 读取数据集元信息，确保标题等内容由主函数控制
    meta = DATASET_META.get(dataset, {})
    title = meta.get("title", dataset.capitalize())

    # 主图宽度为高度的 2.5 倍
    fig, ax = plt.subplots(figsize=(9, 2.5), dpi=600)

    # 采用 SCI 标准的鲜艳配色，点略大于线宽
    line_width = 1.0
    marker_size = 3.5
    ax.plot(
        x,
        y_lambda,
        color="#0072B2",  # bright blue
        marker="o",
        markersize=marker_size,
        markeredgewidth=0.0,
        linewidth=line_width,
        linestyle="-",
        label=r"$\log_{10}(\lambda)$",
    )
    ax.plot(
        x,
        y_mu,
        color="#D55E00",  # bright orange
        marker="s",
        markersize=marker_size,
        markeredgewidth=0.0,
        linewidth=line_width,
        linestyle="--",
        label=r"$\log_{10}(\mu)$",
    )

    # 0 为对称轴，完整展示 [-3, 3]
    ax.set_ylim(-3, 3)
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.2)
    ax.set_yticks(range(-3, 4))

    ax.set_xlabel("times")
    ax.set_ylabel(r"$\log_{10}$ $(\mu,\lambda)$ Value")
    ax.set_xlim(x.min(), x.max())
    # 仅对 (a) Iris 的标题加粗并加大字号，其余使用默认
    _title_text = f"({string.ascii_lowercase[label_idx]}) {title}"
    if dataset.lower() == "iris":
        ax.set_title(_title_text, fontsize=14, fontweight="bold")
    else:
        ax.set_title(_title_text)

    # 坐标刻度与权重均使用默认样式（由 apply_style 控制）

    # 图例置于框内的右上角
    ax.legend(loc="upper right", frameon=True)

    # 保存图像并附带元数据
    out = os.path.join(RES_DIR, f"bayes_best_params_kfold_20times_{dataset}.png")
    fig.canvas.manager.set_window_title(title)
    savefig(fig, out, metadata={"Title": title, "Dataset": dataset})


if __name__ == "__main__":  # pragma: no cover
    # todo 默认绘制的数据集，绘制其他数据集的需要调整
    datasets = sys.argv[1:] or ["iris"]

    # 统一的字体设置：字号变大并加粗
    font_kwargs = {"fontsize": 14, "fontweight": "bold"}

    # 批量绘制多个数据集的参数曲线
    for ds in datasets:
        label_idx = DATASET_LABEL_IDX.get(ds.lower(), 0)
        plot_dataset(ds, label_idx, font_kwargs)
