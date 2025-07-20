# -*- coding: utf-8 -*-
"""静态与动态混合入簇测试脚本
=============================
先按照固定划分初始化 ``MultiClusters``，然后继续按 CSV 顺序动态加入剩余 BBA。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import pandas as pd

# 确保包导入路径指向项目根目录
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 依赖本项目内现成工具函数 / 模块
from cluster.multi_clusters import MultiClusters
from cluster.one_cluster import initialize_empty_cluster
from utility.io import load_bbas
from utility.bba import BBA


def _process_csv_path(argv: List[str], default_csv: str) -> str:
    if argv:
        return argv[0]
    return default_csv


def build_initial_clusters(csv_path: str) -> Tuple[MultiClusters, List[BBA]]:
    """根据 ``DEFAULT_CLUSTERS_ELEMENTS`` 构造初始簇并返回剩余 BBA 列表。"""
    df = pd.read_csv(csv_path)
    bbas, _ = load_bbas(df)

    mc = MultiClusters()
    remaining = []
    for bba in bbas:
        placed = False
        for cname, members in DEFAULT_CLUSTERS_ELEMENTS.items():
            if bba.name in members:
                print(
                    f"------------------------------ Round: {bba.name} ------------------------------"
                )
                if cname not in mc._clusters:
                    clus = initialize_empty_cluster(name=cname)
                    mc.add_cluster(clus)
                else:
                    clus = mc.get_cluster(cname)
                clus.add_bba(bba, _init=True)
                mc.print_all_info()
                placed = True
                break
        if not placed:
            remaining.append(bba)
    return mc, remaining


def continue_dynamic_adding(mc: MultiClusters, bbas: List[BBA]) -> None:
    """继续将剩余 BBA 动态加入 ``mc``。"""
    for bba in bbas:
        print(
            f"------------------------------ Round: {bba.name} ------------------------------"
        )
        mc.add_bba_by_reward(bba)
        mc.print_all_info()


if __name__ == "__main__":  # pragma: no cover
    # todo 在此处更改默认的数据集合
    default_name = "Example_3_7_2.csv"

    # ------------------------------ todo 在此处更改默认起点 ------------------------------
    DEFAULT_CLUSTERS_ELEMENTS: Dict[str, List[str]] = {
        "Clus1": ["m8", "m7", "m6", "m10", "m12", "m13"],
        "Clus2": ["m16", "m15"],
        "Clus3": ["m1", "m2", "m4", "m3"],
    }

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    default_csv = os.path.join(base_dir, "data", "examples", default_name)
    if not os.path.isfile(default_csv):
        print(f"默认 CSV 文件不存在: {default_csv}")
        sys.exit(1)

    csv_path = _process_csv_path(sys.argv[1:], default_csv)

    try:
        print("------------------------------ Initial cluster Here ------------------------------")
        mc, remaining_bbas = build_initial_clusters(csv_path)
        mc.print_all_info()

        print('\n' * 10, end='')
        print("------------------------------ Dynamic BBA Adding Here ------------------------------")
        continue_dynamic_adding(mc, remaining_bbas)

        # visualize_clusters(list(mc._clusters.values()), show=False)
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
