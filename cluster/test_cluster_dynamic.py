# -*- coding: utf-8 -*-
"""动态簇测试脚本
================
验证增量添加 BBA 时簇内计算是否正确。"""

import os
import sys
from typing import Dict, List, Tuple

import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from cluster.one_cluster import initialize_cluster_from_csv, initialize_empty_cluster  # type: ignore
from divergence.bjs import divergence_matrix  # type: ignore
from fractal.fractal_max_entropy import higher_order_bba  # type: ignore
from mean.mean_bba import compute_avg_bba  # type: ignore
from mean.mean_divergence import average_divergence  # type: ignore
from utility.io import load_bbas, parse_focal_set, format_set  # type: ignore


# 根据命令行是否有参数返回簇的csv表与csv路径。
def _process_input_element_list(
        argv: List[str],
        default_clusters: Dict[str, List[str]],
        default_csv: str,
) -> Tuple[Dict[str, List[str]], str]:
    if not argv:
        # 无参数：使用默认簇+CSV
        return default_clusters, default_csv

    # 有参数：`python one_cluster.py <ClusterName> [--csv CSV] BBA1 BBA2 ...`
    cluster_name = argv.pop(0)
    csv_path = default_csv
    if argv and argv[0] == '--csv':
        argv.pop(0)
        if not argv:
            raise ValueError('参数错误: `--csv` 后缺少路径或 BBA 名称')
        csv_path = argv.pop(0)

    bba_names = argv
    if not bba_names:
        raise ValueError('至少指定 1 个 BBA 名称。')

    return {cluster_name: bba_names}, csv_path


# 将不同簇指定的 element list 构建为簇，并且打印簇基本信息。
def construct_clusters_with_dynamic_bba(cluster_element_list: Dict[str, List[str]], csv_path: str) -> None:
    # 读取 CSV 并按文件中顺序获取所有 BBA 名称及其质量
    df = pd.read_csv(csv_path)
    all_bbas, _ = load_bbas(df)  # 返回列表，保持与 CSV 中的顺序
    csv_order_names = [name for name, _ in all_bbas]
    lookup = {name: bba for name, bba in all_bbas}

    # 初始化所有簇
    clusters = {}
    for cname in cluster_element_list:
        clusters[cname] = initialize_empty_cluster(name=cname)

    # 按照 CSV 顺序，将每个 BBA 放入对应簇中
    for bba_name in csv_order_names:
        # 找到该 BBA 所属的簇
        for cname, members in cluster_element_list.items():
            if bba_name in members:
                clus = clusters[cname]
                print(f"Round: {bba_name}")
                clus.add_bba(bba_name, lookup[bba_name])
                clus.print_info()


# ------------------------------ 主函数 ------------------------------ #
if __name__ == '__main__':  # pragma: no cover
    # todo 默认配置，根据不同的CSV 文件或 BBA 簇修改
    example_name = 'Example_3_3.csv'

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    default_csv = os.path.join(base_dir, 'data', 'examples', example_name)
    if not os.path.isfile(default_csv):
        print(f'默认 CSV 文件不存在: {default_csv}')
        sys.exit(1)

    # todo 默认簇名和 BBA 名称，可根据簇的情况修改。
    # 测试用，预设簇及加入顺序（从前到后就是加入顺序）
    default_clusters_elements_list = {
        "Clus1": ["m1", "m2", "m5"],
        "Clus2": ["m3", "m4"],
    }

    try:
        # 解析命令行参数，获取簇规格与 CSV 路径
        cluster_element_list, csv_path = _process_input_element_list(sys.argv[1:], default_clusters_elements_list,
                                                                     default_csv)
        # 模拟将动态进入的BBA放入不同的簇。
        construct_clusters_with_dynamic_bba(cluster_element_list, csv_path)

    except Exception as e:
        # 统一错误输出格式
        print('ERROR:', e)
        sys.exit(1)
