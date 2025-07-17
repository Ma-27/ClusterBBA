# -*- coding: utf-8 -*-
"""多簇操作测试脚本"""

import os
import sys
from typing import List

# 确保包导入路径指向项目根目录
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from cluster.multi_clusters import MultiClusters  # type: ignore
from cluster.visualize_clusters import visualize_clusters
from utility.io import load_bbas  # type: ignore


def _process_csv_path(argv: List[str], default_csv: str) -> str:
    if argv:
        return argv[0]
    return default_csv


# 模拟 BBA 的动态加入
def bba_dynamic_adding(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    bbas, _ = load_bbas(df)
    # 按照 CSV 顺序获取 BBA 名称，模拟动态入簇过程
    mc = MultiClusters()
    for bba in bbas:
        print(f"------------------------------ Round: {bba.name} ------------------------------ ")
        mc.add_bba_by_reward(bba)
        mc.print_all_info()

    # 可视化整个簇集。
    visualize_clusters(list(mc._clusters.values()), show=False)


# ------------------------------ 主函数 ------------------------------ #
if __name__ == '__main__':  # pragma: no cover
    # todo 默认配置，根据不同的 CSV 文件或 BBA 簇修改
    example_name = 'Example_3_3_3.csv'

    # 确定项目根目录：当前脚本位于 cluster/，故上溯一级
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    # 构造数据文件路径（相对于项目根）
    default_csv = os.path.join(base_dir, "data", "examples", example_name)
    if not os.path.isfile(default_csv):
        print(f"默认 CSV 文件不存在: {default_csv}")
        sys.exit(1)

    csv_path = _process_csv_path(sys.argv[1:], default_csv)

    # 构建簇，执行全流程动态入簇。
    try:
        bba_dynamic_adding(csv_path)
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
