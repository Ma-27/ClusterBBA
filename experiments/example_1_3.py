# -*- coding: utf-8 -*-
"""Example 1.3 多元冲突簇划分实验
=================================
根据文档中 Example 1.3 提供的六条 BBA，
验证 ``MultiClusters`` 在高度冲突情况下的分簇效果，
并计算最终簇间的 ``RD_CCJS`` 距离矩阵。
"""

from __future__ import annotations

import os
from typing import Tuple, List

import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from cluster.multi_clusters import MultiClusters
from cluster.one_cluster import Cluster
from divergence.rd_ccjs import divergence_matrix
from mean.mean_divergence import average_divergence
from utility.io import load_bbas


# ------------------------------ 核心流程 ------------------------------ #

def run_experiment(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, float, List[Cluster]]:
    """执行 Example 1.3 分簇实验并返回结果数据"""
    df = pd.read_csv(csv_path)
    bbas, _ = load_bbas(df)

    mc = MultiClusters()
    assignments = []
    for name, bba in bbas:
        target = mc.add_bba_by_reward(name, bba)
        assignments.append([name, target])

    clusters = list(mc._clusters.values())
    dist_df = divergence_matrix(clusters)
    avg_rd = average_divergence(dist_df)

    assign_df = pd.DataFrame(assignments, columns=["BBA", "Cluster"])
    return assign_df, dist_df.round(6), float(avg_rd), clusters


# ------------------------------ 主函数 ------------------------------ #

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "data", "examples", "Example_1_3.csv")
    csv_path = os.path.normpath(csv_path)

    assign_df, dist_df, avg_rd, clusters = run_experiment(csv_path)

    result_dir = os.path.normpath(os.path.join(base_dir, "..", "experiments_result"))
    os.makedirs(result_dir, exist_ok=True)

    assign_df.to_csv(os.path.join(result_dir, "example_1_3_assign.csv"), index=False)
    dist_df.to_csv(os.path.join(result_dir, "example_1_3_rd_ccjs.csv"), float_format="%.6f")

    print(assign_df.to_string(index=False))
    print("\nRD_CCJS 距离矩阵：")
    print(dist_df.to_string())
    print(f"\n平均 RD_CCJS: {avg_rd:.6f}")

    # 可视化簇划分结果
    # visualize_clusters(clusters)
