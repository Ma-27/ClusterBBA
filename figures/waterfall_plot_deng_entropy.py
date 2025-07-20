# -*- coding: utf-8 -*-
"""分形算子下簇心 Deng 熵的瀑布图示例
=======================================
本脚本以 ``waterfall_plot.py`` 为蓝本，读取 Example 3.7 数据集，动态地将 BBA 加入 ``MultiClusters`` 中，并分别在 HOBPA、Average 和 MaxEntropy 三种分形算子下计算簇心 Deng 熵。最终以三维瀑布图展示簇心 Deng 熵随 BBA 数量增加的变化过程。
"""

from __future__ import annotations

import contextlib
import os
import sys
from io import StringIO
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 依赖本项目内现成工具函数 / 模块
from utility.plot_style import apply_style
from utility.formula_labels import LABEL_DENG_ENTROPY
from utility.plot_utils import savefig

apply_style()

from cluster.multi_clusters import MultiClusters  # type: ignore
from entropy.deng_entropy import deng_entropy
from fractal.fractal_hobpa import higher_order_bba as hobpa_fractal
from fractal.fractal_average import higher_order_bba as average_fractal
from fractal.fractal_max_entropy import higher_order_bba as maxent_fractal
from mean.mean_bba import compute_avg_bba
from utility.io import load_bbas  # type: ignore

EntropyHistory = Dict[str, Dict[str, List[float]]]

_fractal_funcs = {
    "HOBPA": hobpa_fractal,
    "Average": average_fractal,
    "MaxEntropy": maxent_fractal,
}


def _centroid_entropy(clus, func):
    """根据指定分形函数计算簇心 Deng 熵。"""
    # 收集簇中所有 BBA
    bbas = clus.get_bbas()
    if not bbas:
        return float("nan")
    # 分形函数需要知道当前最高阶数 h
    h = max(len(bbas) - 1, 0)
    # 对每个 BBA 应用分形算子，得到高阶 BBA
    fbba_list = [func(b, h) for b in bbas]
    # 根据分形后的 BBA 计算平均 BBA 作为簇心
    centroid = compute_avg_bba(fbba_list)
    # 返回簇心的 Deng 熵
    return float(deng_entropy(centroid))


def _record_entropies(step: int, mc: MultiClusters, history: EntropyHistory) -> None:
    """记录当前簇心在不同分形下的 Deng 熵。"""
    clusters = list(mc._clusters.values())
    # 遍历每个簇，计算其在各分形算子下的簇心 Deng 熵
    for clus in clusters:
        for name, func in _fractal_funcs.items():
            # 对应簇在此前若未出现，需要填补前面步长的 NaN
            history.setdefault(name, {}).setdefault(clus.name, [float("nan")] * (step - 1))
            ent = _centroid_entropy(clus, func)
            history[name][clus.name].append(ent)
    # 若某簇在本轮已经消失，仍需补齐记录长度
    for key in history:
        for cname, vals in history[key].items():
            if len(vals) < step:
                vals.append(float("nan"))


def _plot_history(history: EntropyHistory, n_steps: int, out_path: str) -> None:
    """根据记录绘制三维瀑布图"""
    alg_names = list(_fractal_funcs.keys())
    y_pos = np.arange(len(alg_names))
    # 收集所有出现过的簇名称
    all_names: set[str] = set()
    for key in history:
        all_names.update(history[key].keys())
    cluster_names = sorted(all_names)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # constrained_layout 能更好地处理三维图的边距问题，避免 ``tight_layout``
    # 带来的无法收敛警告
    fig = plt.figure(figsize=(6, 4), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect((2, 1, 1))

    ax.set_xlabel('Step')
    ax.set_ylabel('')
    ax.set_zlabel(LABEL_DENG_ENTROPY)

    ax.set_xlim(1, n_steps)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(alg_names)

    # 逐簇绘制瀑布曲线
    for idx, cname in enumerate(cluster_names):
        color = colors[idx % len(colors)]
        for a_idx, alg in enumerate(alg_names):
            # 获取指定算法下的熵序列
            vals = history.get(alg, {}).get(cname, [])
            x_vals = np.arange(1, len(vals) + 1)
            y_val = y_pos[a_idx]
            y_points = np.full_like(x_vals, y_val)
            ax.plot(x_vals, y_points, vals, color=color, linewidth=1)
            ax.scatter(x_vals, y_points, vals, color=color, s=10)
            # 封闭曲面以形成瀑布效果
            top = list(zip(x_vals, y_points, vals))
            bottom = list(zip(x_vals, y_points, np.zeros_like(vals)))
            poly = Poly3DCollection([top + bottom[::-1]], alpha=0.15)
            poly.set_color(color)
            ax.add_collection3d(poly)

    # 构造代理句柄显示簇名称
    legend_proxies = [
        plt.Line2D([0], [0], color=colors[i % len(colors)], lw=1.5)
        for i in range(len(cluster_names))
    ]
    ax.legend(legend_proxies, cluster_names, loc="upper left", bbox_to_anchor=(0, 1.0))
    # 调整视角以获得更好的瀑布效果
    ax.view_init(elev=20, azim=-60)
    savefig(fig, out_path)


if __name__ == '__main__':  # pragma: no cover
    # todo 默认使用的示例数据集，可根据需要修改
    example_name = 'Example_3_7.csv'
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    default_csv = os.path.join(base_dir, 'data', 'examples', example_name)
    if not os.path.isfile(default_csv):
        print(f'默认 CSV 文件不存在: {default_csv}')
        sys.exit(1)

    # 读取 CSV 并逐条加入 ``MultiClusters``
    df = pd.read_csv(default_csv)
    bbas, _ = load_bbas(df)
    mc = MultiClusters()
    history: EntropyHistory = {}
    step = 0
    for bba in bbas:
        step += 1
        # 屏蔽 ``add_bba_by_reward`` 的控制台输出
        with contextlib.redirect_stdout(StringIO()):
            mc.add_bba_by_reward(bba)
        _record_entropies(step, mc, history)

    # 输出目录，统一位于 ``experiments_result``
    res_dir = os.path.normpath(os.path.join(base_dir, 'experiments_result'))
    os.makedirs(res_dir, exist_ok=True)

    dataset = os.path.splitext(os.path.basename(example_name))[0]
    suffix = dataset.lower()
    if suffix.startswith('example_'):
        suffix = suffix[len('example_'):]
    out_path = os.path.join(res_dir, f'waterfall_deng_entropy_{suffix}.png')

    _plot_history(history, step, out_path)
    print(f'Figure saved to: {out_path}')
