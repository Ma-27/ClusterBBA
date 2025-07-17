# -*- coding: utf-8 -*-
"""快速打印簇划分结果的脚本
=========================
读取示例 CSV 文件，使用 :class:`MultiClusters` 自动分簇，
最后仅打印每个簇包含哪些 BBA 名称。
"""

from __future__ import annotations

import os
import random
import sys
from typing import List

# 确保包导入路径指向项目根目录
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from cluster.multi_clusters import (
    construct_clusters_by_sequence,
    construct_clusters_by_sequence_dp,
)  # type: ignore
from utility.io import load_bbas, ensure_focal_order  # type: ignore


def _process_csv_path(argv: List[str], default_csv: str) -> str:
    if argv:
        return argv[0]
    return default_csv


def print_cluster_elements(csv_path: str, use_dp: bool = False) -> None:
    df = pd.read_csv(csv_path)
    bbas, _ = load_bbas(df)

    # 使用构建接口批量加入 BBA
    if use_dp:
        random.shuffle(bbas)
        order = ensure_focal_order(bbas, None)
        rows = [[bba.name] + bba.to_series(order) for bba in bbas]
        df_bba = pd.DataFrame(rows, columns=["BBA"] + order)
        print(df_bba.to_markdown(tablefmt="github", floatfmt=".4f"))
        mc = construct_clusters_by_sequence_dp(bbas)
    else:
        mc = construct_clusters_by_sequence(bbas)

    clusters = mc._clusters
    print(f"Number of clusters: {len(clusters)}")
    for cname, clus in clusters.items():
        elems = ", ".join(b.name for b in clus.get_bbas())
        print(f"Cluster '{cname}' Elements: {elems}")


if __name__ == "__main__":  # pragma: no cover
    # todo 默认配置，根据不同的 CSV 文件或 BBA 簇修改
    example_name = "Example_3_7.csv"

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    default_csv = os.path.join(base_dir, "data", "examples", example_name)
    if not os.path.isfile(default_csv):
        print(f"默认 CSV 文件不存在: {default_csv}")
        sys.exit(1)

    # todo 你可以选择是否启用动态规划，原始研究中是不启用的，但是启用了之后，分簇结果会好很多。
    # 在添加 BBA 的选簇策略时，是否应该使用动态规划。动态规划搜索的是全局最优解，能克服在线贪心收益计算中，分簇不稳定的问题。
    use_dp = False
    args = sys.argv[1:]
    if "--dp" in args:
        use_dp = True
        args.remove("--dp")

    csv_path = _process_csv_path(args, default_csv)
    try:
        print_cluster_elements(csv_path, use_dp=use_dp)
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
