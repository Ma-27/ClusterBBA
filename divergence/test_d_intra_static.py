# -*- coding: utf-8 -*-
"""test_d_intra_static.py

静态指定簇内 BBA，计算 ``D_intra`` 随时间的变化。

该脚本与 ``test_d_intra.py`` 功能相同，但簇划分由用户手动给定，
不再根据收益动态选择簇。
"""

import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

# 依赖本项目内现成工具函数 / 模块
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from cluster.one_cluster import Cluster, initialize_empty_cluster  # type: ignore
from cluster.multi_clusters import MultiClusters  # type: ignore
from utility.formula_labels import LABEL_D_INTRA
from utility.plot_style import apply_style
from utility.plot_utils import savefig
from utility.io import load_bbas  # type: ignore

apply_style()

DIHistory = Dict[str, List[float]]


def _calc_intra(clus: Cluster, clusters: List[Cluster], handle_boundary: bool) -> float:
    """使用 ``MultiClusters`` 的逻辑计算 ``D_intra``。"""
    val = MultiClusters._calc_intra_divergence(clus, clusters, handle_boundary)
    if val is None:
        return float('nan')
    return float(val)


def _record_history(step: int, clusters: Dict[str, Cluster], history: DIHistory, handle_boundary: bool) -> None:
    """记录当前各簇的 ``D_intra``。"""
    clus_list = list(clusters.values())
    for clus in clus_list:
        val = _calc_intra(clus, clus_list, handle_boundary)
        if clus.name not in history:
            history[clus.name] = [float('nan')] * (step - 1)
        history[clus.name].append(val)
    for cname, vals in history.items():
        if len(vals) < step:
            vals.append(float('nan'))

    print(f"Step {step} D_intra:")
    for clus in clus_list:
        di_val = history[clus.name][-1]
        if di_val != di_val:
            print(f"  {clus.name}: None")
        else:
            est_flag = handle_boundary and len(clus.get_bbas()) == 1
            star = "*" if est_flag else ""
            print(f"  {clus.name}: {di_val:.4f}{star}")
    print()


def _plot_history(history: DIHistory, save_path: str | None = None, show: bool = True) -> None:
    """绘制 ``D_intra`` 随时间变化的折线图。"""
    steps = range(1, max(len(v) for v in history.values()) + 1)
    for cname, vals in history.items():
        plt.plot(steps, vals, marker='o', label=cname)
    plt.xlabel('Step')
    plt.ylabel(LABEL_D_INTRA)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()
    if save_path:
        savefig(save_path)
    else:
        savefig('d_intra_history.png')
    if show:
        plt.show()


if __name__ == "__main__":
    # todo 默认示例文件名，可按需要修改
    default_name = "Example_3_7_2.csv"

    # todo 默认簇及其包含的 BBA 名称
    default_clusters_elements = {
        "Clus1": ["m8", "m7", "m6", "m10", "m12", "m13"],
        "Clus2": ["m16", "m15", "m14"],
        "Clus3": ["m1", "m2", "m4", "m3"],
    }

    # 命令行格式：python test_d_intra_static.py [CSV] [--handle-boundary]
    args = sys.argv[1:]

    # todo 默认使用替代策略，评估替代策略是否合理
    handle_boundary = True
    csv_name = default_name
    for arg in args:
        if arg in ("-b", "--handle-boundary"):
            handle_boundary = True
        else:
            csv_name = arg

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, "..", "data", "examples", csv_name)
    if not os.path.isfile(csv_path):
        print(f"找不到 CSV 文件: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    all_bbas, _ = load_bbas(df)

    # 初始化簇
    clusters: Dict[str, Cluster] = {
        cname: initialize_empty_cluster(name=cname)
        for cname in default_clusters_elements
    }

    history: DIHistory = {}
    step = 0

    # 按 CSV 顺序将 BBA 加入对应簇，并记录 ``D_intra``
    for bba in all_bbas:
        for cname, members in default_clusters_elements.items():
            if bba.name in members:
                clusters[cname].add_bba(bba)
                break
        step += 1
        _record_history(step, clusters, history, handle_boundary)

    clus_list = list(clusters.values())

    print("\n----- D_intra per Cluster -----")
    results = []
    for clus in clus_list:
        di = MultiClusters._calc_intra_divergence(clus, clus_list, handle_boundary)
        if di is None:
            print(f"{clus.name}: None")
            results.append([None])
        else:
            est_flag = handle_boundary and len(clus.get_bbas()) == 1
            star = "*" if est_flag else ""
            print(f"{clus.name}: {di:.4f}{star}")
            results.append([f"{di:.4f}{star}"])

    df_res = pd.DataFrame(results, index=[c.name for c in clus_list], columns=["D_intra"])

    out_dir = os.path.abspath(os.path.join(base, "..", "experiments_result"))
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"d_intra_{os.path.splitext(csv_name)[0]}.csv")
    df_res.to_csv(out_csv, float_format="%.4f", index_label="Cluster")
    print(f"结果 CSV: experiments_result/{os.path.basename(out_csv)}")

    dataset = os.path.splitext(os.path.basename(csv_name))[0]
    suffix = dataset.lower()
    if suffix.startswith('example_'):
        suffix = suffix[len('example_'):]
    fig_path = os.path.join(out_dir, f'example_{suffix}_d_intra_history.png')
    _plot_history(history, save_path=fig_path)
    print(f'History figure saved to: {fig_path}')
