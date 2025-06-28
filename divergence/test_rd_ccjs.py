# -*- coding: utf-8 -*-
"""test_rd_ccjs.py

基于 ``cluster`` 动态加簇过程，计算簇与簇之间的 RD_CCJS 距离矩阵并验证其度量性质。
"""

import os
import sys
from typing import Dict, List

import pandas as pd

from cluster import initialize_empty_cluster  # type: ignore
from divergence.rd_ccjs import metric_matrix, save_csv  # type: ignore
from divergence.metric_test import (
    test_nonnegativity,
    test_symmetry,
    test_triangle_inequality,
)  # type: ignore
from utility.io import load_bbas  # type: ignore


# 根据 Example_3_3.csv 的顺序动态加入簇
DEFAULT_ASSIGNMENT: Dict[str, List[str]] = {
    "Clus1": ["m1", "m2", "m5"],
    "Clus2": ["m3", "m4"],
}


if __name__ == "__main__":
    default_name = "Example_3_3.csv"
    csv_name = sys.argv[1] if len(sys.argv) > 1 else default_name

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, "..", "data", "examples", csv_name)
    if not os.path.isfile(csv_path):
        print(f"找不到 CSV 文件: {csv_path}")
        sys.exit(1)

    # 读取 BBA
    df = pd.read_csv(csv_path)
    all_bbas, _ = load_bbas(df)
    order_names = [name for name, _ in all_bbas]
    lookup = {name: bba for name, bba in all_bbas}

    # 初始化簇
    clusters = {name: initialize_empty_cluster(name) for name in DEFAULT_ASSIGNMENT}

    # 按 CSV 顺序动态添加
    for bba_name in order_names:
        for c_name, members in DEFAULT_ASSIGNMENT.items():
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

    # 度量检验
    test_symmetry(dist_df)
    test_nonnegativity(dist_df)
    test_triangle_inequality(dist_df)
