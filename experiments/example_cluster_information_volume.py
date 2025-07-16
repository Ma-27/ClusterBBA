# -*- coding: utf-8 -*-
"""Dynamic clustering and Information Volume
============================================
根据 `test_multi_clusters.py` 的动态入簇流程，将 CSV 中的 BBA 顺序加入 :class:`MultiClusters`，
再对每个簇心执行 :func:`entropy.information_volume.information_volume`,计算信息体积并绘制 ``Hi(m)`` 收敛曲线。

可在命令行指定 CSV 文件名以替换默认 ``Example_3_3_3.csv``。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from cluster.multi_clusters import MultiClusters
from entropy.information_volume import information_volume
from utility.formula_labels import LABEL_I, LABEL_HI_M
from utility.io import load_bbas
from utility.plot_style import apply_style
from utility.plot_utils import savefig

apply_style()


def dynamic_clustering(csv_path: str) -> MultiClusters:
    """按 CSV 顺序将 BBA 动态加入簇中并返回 :class:`MultiClusters`."""
    df = pd.read_csv(csv_path)
    # 解析出所有 BBA
    bbas, _ = load_bbas(df)
    mc = MultiClusters()
    for name, bba in bbas:
        # 每轮将新的 BBA 加入并输出簇信息
        print(f"------------------------------ Round: {name} ------------------------------")
        mc.add_bba_by_reward(name, bba)
        mc.print_all_info()
    return mc


def compute_information_volume(mc: MultiClusters) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    """对每个簇心计算信息体积并返回结果表与曲线数据"""
    records: List[Tuple[str, float]] = []
    curves: Dict[str, List[float]] = {}
    print("----- Information Volume 结果 -----")
    for name, clus in mc._clusters.items():
        centroid = clus.get_centroid()
        if centroid is None:
            continue
        # 计算簇心的信息体积及其收敛曲线
        vol, curve = information_volume(centroid, return_curve=True)
        print(f"\n{name}:")
        for i, h in enumerate(curve, start=1):
            print(f"  i={i:<2d} Hi(m)={h:.6f}")
        print(f"  >>> HIV-mass(m) = {vol:.6f}")
        records.append((name, round(vol, 6)))
        curves[name] = curve
    df = pd.DataFrame(records, columns=["Cluster", "InformationVolume"])
    return df, curves


def plot_iv_curves(curves: Dict[str, List[float]], out_path: str) -> None:
    """绘制 Hi(m) 收敛曲线，不检查重合线"""
    fig, ax = plt.subplots()
    for name, vals in curves.items():
        steps = list(range(1, len(vals) + 1))
        ax.plot(steps, vals, label=name)
    ax.set_xlabel(LABEL_I)
    ax.set_ylabel(LABEL_HI_M)
    ax.set_title("Information Volume Convergence")
    ax.legend()
    savefig(fig, out_path)


if __name__ == "__main__":
    # todo 默认示例文件名，可根据实际情况修改
    default_name = "Example_3_3_3.csv"
    csv_name = sys.argv[1] if len(sys.argv) > 1 else default_name

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "data", "examples", csv_name)
    csv_path = os.path.normpath(csv_path)
    if not os.path.isfile(csv_path):
        print(f"找不到 CSV 文件: {csv_path}")
        sys.exit(1)

    mc = dynamic_clustering(csv_path)

    df, curves = compute_information_volume(mc)

    result_dir = os.path.normpath(os.path.join(base_dir, "..", "experiments_result"))
    # 保存信息体积实验的输出
    os.makedirs(result_dir, exist_ok=True)
    dataset = os.path.splitext(os.path.basename(csv_name))[0]
    csv_out = os.path.join(result_dir, f"cluster_information_volume_{dataset}.csv")
    df.to_csv(csv_out, index=False, float_format="%.6f")

    fig_out = os.path.join(result_dir, f"cluster_information_volume_{dataset}_curve.png")
    plot_iv_curves(curves, fig_out)

    print(f"Results saved to: {csv_out}")
    print(f"Figure saved to: {fig_out}")
