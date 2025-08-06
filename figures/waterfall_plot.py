# -*- coding: utf-8 -*-
"""Example 3.3.3 三指标面墙示例
===============================
仿照 ``group_surface.py``，在 Example 3.3.3 数据集上，动态加入 BBA 后计算每个簇心的 Deng 熵、各簇的簇内散度，以及簇与簇之间的 RD_CCJS 距离，并以三面墙展示。

本版本对原始绘图进行改进，使得 ``D_intra``、``RD_CCJS`` 与``Deng`` 熵各自采用独立的纵轴范围，从而在数值量级差异较大时也能清楚地观察其变化趋势。RD_CCJS 面墙的绘制方式亦进行了调整：每一对簇之间的距离将单独形成一个瀑布曲线，因此若存在 ``K`` 个簇，则会绘制 ``K(K-1)/2`` 条瀑布曲线。
"""

from __future__ import annotations

import contextlib
import os
import sys
from io import StringIO
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 确保包导入路径指向项目根目录
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 依赖本项目内现成工具函数 / 模块
from utility.plot_style import apply_style
from utility.formula_labels import LABEL_RD_CCJS, LABEL_DENG_ENTROPY, LABEL_D_INTRA
from utility.plot_utils import savefig

apply_style()

from cluster.multi_clusters import MultiClusters  # type: ignore
from entropy.deng_entropy import deng_entropy
from divergence.rd_ccjs import divergence_matrix
from utility.io import load_bbas  # type: ignore
from config import INTRA_EPS

MetricHistory = Dict[str, Dict[str, List[float]]]


def _calc_intra(clus, clusters):
    """使用 ``MultiClusters`` 的 API 计算 ``D_intra``。"""
    val = MultiClusters._calc_intra_divergence(clus, clusters)
    if val is None:
        return float(INTRA_EPS)
    return float(val)


def _record_metrics(step: int, mc: MultiClusters, history: MetricHistory) -> None:
    """记录当前各簇指标。"""
    clusters = list(mc._clusters.values())
    dist_df: Optional[pd.DataFrame] = None
    # 当簇数量大于 1 时计算 RD_CCJS 距离矩阵
    if len(clusters) >= 2:
        dist_df = divergence_matrix(clusters)

    # 逐簇记录熵与簇内散度
    for clus in clusters:
        for key in ("entropy", "d_intra"):
            history.setdefault(key, {}).setdefault(clus.name, [float("nan")] * (step - 1))
        ent = deng_entropy(clus.get_centroid() or {})
        di = _calc_intra(clus, clusters)
        history["entropy"][clus.name].append(ent)
        history["d_intra"][clus.name].append(di)

    # 补齐本轮缺失的簇
    for key in ("entropy", "d_intra"):
        for name, vals in history[key].items():
            if len(vals) < step:
                vals.append(float("nan"))

    # 记录每一对簇之间的 RD_CCJS
    if dist_df is not None:
        names = list(dist_df.columns)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pair = f"{names[i]}-{names[j]}"
                history.setdefault("rd_ccjs", {}).setdefault(pair, [float("nan")] * (step - 1))
                dist = float(dist_df.loc[names[i], names[j]])
                history["rd_ccjs"][pair].append(dist)

    # 填充本轮未出现的 pair
    if "rd_ccjs" in history:
        for pair, vals in history["rd_ccjs"].items():
            if len(vals) < step:
                vals.append(float("nan"))


def _plot_history(history: MetricHistory, n_steps: int, out_path: str) -> None:
    """将历史指标绘制为三面墙，并为每个指标自适应纵轴范围"""

    metrics = ["entropy", "d_intra", "rd_ccjs"]
    metric_labels = [LABEL_DENG_ENTROPY, LABEL_D_INTRA, LABEL_RD_CCJS]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # 创建宽屏布局以容纳三个子图
    # 调整为更宽、更扁的画布
    fig = plt.figure(figsize=(14, 4))
    # 适当增大横向间隔，减小上下留白
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.06, top=0.94, wspace=0.4)

    # 分别在三个子图中绘制三种指标
    for m_idx, metric in enumerate(metrics, start=1):
        ax = fig.add_subplot(1, 3, m_idx, projection='3d')
        ax.set_box_aspect((2, 1, 1))

        ax.set_xlabel('Step')
        if metric == "rd_ccjs":
            ax.set_ylabel('Pair')
        else:
            ax.set_ylabel('Clus')
        ax.set_title(metric_labels[m_idx - 1])
        ax.set_xlim(1, n_steps)

        names = sorted(history.get(metric, {}).keys())
        y_pos = np.arange(len(names))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, rotation=30)

        # 计算当前指标的取值范围以确定 z 轴上下界
        vals_all: List[float] = []
        for cname in names:
            vals_all.extend(history.get(metric, {}).get(cname, []))
        valid_vals = [v for v in vals_all if not np.isnan(v)]
        if valid_vals:
            z_min = min(valid_vals)
            z_max = max(valid_vals)
        else:
            z_min, z_max = 0.0, 1.0
        if z_min == z_max:
            z_max = z_min + 1.0
        ax.set_zlim(z_min, z_max)
        ax.set_zlabel('Value')

        for idx, cname in enumerate(names):
            color = colors[idx % len(colors)]
            # 单个簇在不同时间步的数值
            vals = history.get(metric, {}).get(cname, [])
            x_vals = np.arange(1, len(vals) + 1)
            y_val = y_pos[idx]
            y_points = np.full_like(x_vals, y_val)
            ax.plot(x_vals, y_points, vals, color=color, linewidth=1)
            ax.scatter(x_vals, y_points, vals, color=color, s=10)
            # 使用半透明多边形形成“瀑布”效果
            top = list(zip(x_vals, y_points, vals))
            bottom = list(zip(x_vals, y_points, np.full_like(vals, z_min)))
            poly = Poly3DCollection([top + bottom[::-1]], alpha=0.15)
            poly.set_color(color)
            ax.add_collection3d(poly)

        if m_idx == 1:
            # 仅在第一幅子图上添加图例并放置在图外侧
            legend_proxies = [
                plt.Line2D([0], [0], color=colors[i % len(colors)], lw=1.5)
                for i in range(len(names))
            ]
            ax.legend(
                legend_proxies,
                names,
                loc='upper left',
                bbox_to_anchor=(-0.1, 1.02),
                borderaxespad=0.0,
            )
        # 调整视角以提升可读性
        ax.view_init(elev=20, azim=-60)

    # 使用统一的保存函数导出图像
    savefig(fig, out_path)


if __name__ == '__main__':  # pragma: no cover
    # todo 在这里更改数据集，对其他的进行演示。
    example_name = 'Example_3_7.csv'
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    default_csv = os.path.join(base_dir, 'data', 'examples', example_name)
    if not os.path.isfile(default_csv):
        print(f'默认 CSV 文件不存在: {default_csv}')
        sys.exit(1)

    df = pd.read_csv(default_csv)
    bbas, _ = load_bbas(df)
    mc = MultiClusters()
    history: MetricHistory = {}
    step = 0
    for bba in bbas:
        step += 1
        with contextlib.redirect_stdout(StringIO()):
            mc.add_bba_by_reward(bba)
        _record_metrics(step, mc, history)

    res_dir = os.path.normpath(os.path.join(base_dir, 'experiments_result'))
    os.makedirs(res_dir, exist_ok=True)

    dataset = os.path.splitext(os.path.basename(example_name))[0]
    suffix = dataset.lower()
    if suffix.startswith('example_'):
        suffix = suffix[len('example_'):]
    # 图像保存路径
    out_path = os.path.join(res_dir, f'waterfall_plot_example_{suffix}.png')

    _plot_history(history, step, out_path)
    print(f'Figure saved to: {out_path}')
