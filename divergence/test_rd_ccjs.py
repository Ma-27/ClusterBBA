# -*- coding: utf-8 -*-
"""test_rd_ccjs.py

基于 ``cluster`` 动态加簇过程，计算簇与簇之间的 RD_CCJS 距离矩阵并验证其度量性质。
"""

import os
import sys
from typing import Dict, List

import pandas as pd

from cluster.one_cluster import initialize_empty_cluster  # type: ignore
from divergence.metric_test import (
    test_nonnegativity,
    test_symmetry,
    test_triangle_inequality,
)  # type: ignore
from divergence.rd_ccjs import metric_matrix, save_csv  # type: ignore
from utility.io import load_bbas  # type: ignore

if __name__ == "__main__":
    # todo 默认示例文件名，可根据实际情况修改
    default_name = "Example_1_2.csv"
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
        "Clus1": ["m1"],
        "Clus2": ["m2"],
    }

    # 初始化簇
    clusters = {name: initialize_empty_cluster(name) for name in DEFAULT_CLUSTER_ASSIGNMENT}

    # 按 CSV 顺序动态添加 BBAs
    for bba_name in order_names:
        # 遍历每个簇，找到该 BBA 所属的簇并添加
        for c_name, members in DEFAULT_CLUSTER_ASSIGNMENT.items():
            if bba_name in members:
                clusters[c_name].add_bba(bba_name, lookup[bba_name])

    clus_list = list(clusters.values())

    # 计算 RD_CCJS 距离矩阵
    dist_df = metric_matrix(clus_list)

    print("\n----- RD_CCJS Metric Matrix -----")
    print(dist_df.to_string())

    save_csv(dist_df, default_name=csv_name, label="metric")
    print(
        f"结果 CSV: experiments_result/rd_ccjs_metric_{os.path.splitext(csv_name)[0]}.csv"
    )

    # 1. 测试对称性
    test_symmetry(dist_df)

    # 2. 测试非负性
    test_nonnegativity(dist_df)

    # 3. 测试三角不等式
    test_triangle_inequality(dist_df)
