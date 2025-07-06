# -*- coding: utf-8 -*-
"""快速打印簇划分结果的脚本
=========================
读取示例 CSV 文件，使用 :class:`MultiClusters` 自动分簇，
最后仅打印每个簇包含哪些 BBA 名称。
"""

from __future__ import annotations

import os
import sys
from typing import List

# 确保包导入路径指向项目根目录
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import io
import pandas as pd
from contextlib import redirect_stdout

# 依赖本项目内现成工具函数 / 模块
from cluster.multi_clusters import MultiClusters  # type: ignore
from utility.io import load_bbas  # type: ignore


def _process_csv_path(argv: List[str], default_csv: str) -> str:
    if argv:
        return argv[0]
    return default_csv


def print_cluster_elements(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    bbas, _ = load_bbas(df)

    mc = MultiClusters()
    for name, bba in bbas:
        with redirect_stdout(io.StringIO()):
            mc.add_bba_by_reward(name, bba)

    clusters = mc._clusters
    print(f"Number of clusters: {len(clusters)}")
    for cname, clus in clusters.items():
        elems = ", ".join(name for name, _ in clus.get_bbas())
        print(f"Cluster '{cname}' Elements: {elems}")


if __name__ == "__main__":  # pragma: no cover
    # todo 默认配置，根据不同的 CSV 文件或 BBA 簇修改
    example_name = "Example_3_1.csv"

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    default_csv = os.path.join(base_dir, "data", "examples", example_name)
    if not os.path.isfile(default_csv):
        print(f"默认 CSV 文件不存在: {default_csv}")
        sys.exit(1)

    csv_path = _process_csv_path(sys.argv[1:], default_csv)
    try:
        print_cluster_elements(csv_path)
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
