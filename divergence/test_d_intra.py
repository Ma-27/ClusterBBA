# -*- coding: utf-8 -*-
"""test_d_intra.py

基于 ``cluster`` 动态加簇过程，计算每个簇的 ``D_intra``。
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


def _calc_intra(clus: Cluster, clusters: List[Cluster]) -> float:
    """使用 ``MultiClusters`` 的逻辑计算 ``D_intra``。"""
    val = MultiClusters._calc_intra_divergence(clus, clusters)
    if val is None:
        return float('nan')
    return float(val)


def _record_history(step: int, clusters: Dict[str, Cluster], history: DIHistory) -> None:
    """记录当前各簇的 ``D_intra``。"""
    clus_list = list(clusters.values())
    for clus in clus_list:
        val = _calc_intra(clus, clus_list)
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
            print(f"  {clus.name}: {di_val:.4f}")
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
    # todo 默认示例文件名，可根据实际情况修改
    default_name = "Example_3_7.csv"
    # 处理命令行参数：CSV 文件名
    csv_name = sys.argv[1] if len(sys.argv) > 1 else default_name

    # 确定项目根目录：当前脚本位于 divergence/，故上溯一级
    base = os.path.dirname(os.path.abspath(__file__))
    # 构造数据文件路径（相对于项目根）
    csv_path = os.path.join(base, "..", "data", "examples", csv_name)
    if not os.path.isfile(csv_path):
        print(f"找不到 CSV 文件: {csv_path}")
        sys.exit(1)

    # 读取 BBA
    df = pd.read_csv(csv_path)
    all_bbas, _ = load_bbas(df)
    order_names = [b.name for b in all_bbas]
    lookup = {b.name: b for b in all_bbas}

    # todo 这里硬性指定簇的名称和成员列表，请根据数据集对应的实际情况修改
    DEFAULT_CLUSTER_ASSIGNMENT: Dict[str, List[str]] = {
        "Clus1": ["m1", "m2", "m3", "m4", "m5"],
        "Clus2": ["m6", "m7", "m8", "m9", "m10", "m11", "m12", "m13"],
        "Clus3": ["m14", "m15", "m16"],
    }

    # 初始化簇
    clusters = {name: initialize_empty_cluster(name) for name in DEFAULT_CLUSTER_ASSIGNMENT}

    history: DIHistory = {}
    step = 0

    # 按 CSV 顺序动态添加 BBAs
    for bba_name in order_names:
        step += 1
        for c_name, members in DEFAULT_CLUSTER_ASSIGNMENT.items():
            if bba_name in members:
                clusters[c_name].add_bba(lookup[bba_name])
        _record_history(step, clusters, history)

    clus_list = list(clusters.values())

    print("\n----- D_intra per Cluster -----")
    results = []
    for clus in clus_list:
        di = clus.intra_divergence()
        if di is None:
            print(f"{clus.name}: None")
            results.append([None])
        else:
            print(f"{clus.name}: {di:.4f}")
            results.append([di])

    df_res = pd.DataFrame(results, index=[c.name for c in clus_list], columns=["D_intra"])

    # 保存结果 CSV
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
