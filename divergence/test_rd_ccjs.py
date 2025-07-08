# -*- coding: utf-8 -*-
"""test_rd_ccjs.py

基于 ``cluster`` 动态加簇过程，计算簇与簇之间的 RD_CCJS 距离矩阵并验证其度量性质。
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
from divergence.metric_test import (
    test_nonnegativity,
    test_symmetry,
    test_triangle_inequality,
)  # type: ignore
from divergence.rd_ccjs import divergence_matrix, save_csv  # type: ignore
from mean.mean_divergence import average_divergence  # type: ignore
from utility.formula_labels import LABEL_RD_CCJS
from utility.plot_style import apply_style
from utility.plot_utils import savefig
from utility.io import load_bbas  # type: ignore

apply_style()

RDHistory = List[float]


def _record_history(step: int, clusters: Dict[str, 'Cluster'], history: RDHistory) -> None:
    """记录整个簇集的平均 ``RD_CCJS``。"""
    clus_list = list(clusters.values())
    dist_df = divergence_matrix(clus_list)
    avg = average_divergence(dist_df)
    history.append(avg)
    print(f"Step {step} Avg RD_CCJS: {avg:.4f}")


def _plot_history(history: RDHistory, save_path: str | None = None, show: bool = True) -> None:
    """绘制平均 ``RD_CCJS`` 随时间变化的折线图。"""
    steps = range(1, len(history) + 1)
    plt.plot(list(steps), history, marker='o', label=f'Average {LABEL_RD_CCJS}')
    plt.xlabel('Step')
    plt.ylabel(LABEL_RD_CCJS)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()
    if save_path:
        savefig(save_path)
    else:
        savefig('rd_ccjs_history.png')
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
    order_names = [name for name, _ in all_bbas]
    lookup = {name: bba for name, bba in all_bbas}

    # todo 这里硬性指定簇的名称和成员列表，请根据数据集对应的实际情况修改
    DEFAULT_CLUSTER_ASSIGNMENT: Dict[str, List[str]] = {
        "Clus1": ["m1", "m2", "m5", "m6"],
        "Clus2": ["m3", "m4"],
        "Clus3": ["m7"],
    }

    # 初始化簇
    clusters = {name: initialize_empty_cluster(name) for name in DEFAULT_CLUSTER_ASSIGNMENT}

    history: RDHistory = []
    step = 0

    # 按 CSV 顺序动态添加 BBAs
    for bba_name in order_names:
        step += 1
        for c_name, members in DEFAULT_CLUSTER_ASSIGNMENT.items():
            if bba_name in members:
                clusters[c_name].add_bba(bba_name, lookup[bba_name])
        _record_history(step, clusters, history)

    clus_list = list(clusters.values())

    # 计算 RD_CCJS 距离矩阵
    dist_df = divergence_matrix(clus_list)

    print("\n----- RD_CCJS Metric Matrix -----")
    print(dist_df.to_string())

    save_csv(dist_df, default_name=csv_name, label="divergence")
    print(
        f"结果 CSV: experiments_result/rd_ccjs_divergence_{os.path.splitext(csv_name)[0]}.csv"
    )

    # 1. 测试对称性
    test_symmetry(dist_df)

    # 2. 测试非负性
    test_nonnegativity(dist_df)

    # 3. 测试三角不等式
    test_triangle_inequality(dist_df)

    # 绘制平均 RD_CCJS 变化历史
    res_dir = os.path.abspath(os.path.join(base, '..', 'experiments_result'))
    os.makedirs(res_dir, exist_ok=True)
    dataset = os.path.splitext(os.path.basename(csv_name))[0]
    suffix = dataset.lower()
    if suffix.startswith('example_'):
        suffix = suffix[len('example_'):]
    fig_path = os.path.join(res_dir, f'example_{suffix}_rd_ccjs_history.png')

    _plot_history(history, save_path=fig_path)
    print(f'History figure saved to: {fig_path}')
