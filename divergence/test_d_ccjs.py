# -*- coding: utf-8 -*-
"""
test_d_ccjs.py

使用 divergence.ccjs 包计算 CCJS 距离矩阵，支持传入超参数 n，并验证度量性质
"""

import os
import sys

import pandas as pd

from divergence.d_ccjs import metric_matrix, save_csv, divergence_matrix  # type: ignore
from divergence.metric_test import test_nonnegativity, test_symmetry, test_triangle_inequality  # type: ignore
# 依赖本项目内现成工具函数 / 模块
from utility.io import load_bbas  # type: ignore

# ------------------------------ 主函数 ------------------------------ #
if __name__ == '__main__':
    # todo 默认示例文件名，可根据实际情况修改
    default_name = 'Example_0_4.csv'

    # 支持通过命令行参数指定超参数 n
    csv_name = sys.argv[1] if len(sys.argv) > 1 else default_name

    # 确定项目根目录：当前脚本位于 divergence/，故上溯一级
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # 构造数据文件路径（相对于项目根）
    csv_path = os.path.join(project_root, 'data', 'examples', csv_name)
    if not os.path.isfile(csv_path):
        print(f"找不到 CSV 文件: {csv_path}")
        sys.exit(1)

    # 读取 CSV 并解析 BBA 数据
    df = pd.read_csv(csv_path)
    bbas, _ = load_bbas(df)

    # todo 构造簇规模参数：请手动指定各簇的规模（根据实际 bbas 名称修改）
    # 例如，若有三个簇，名称分别为 'clus1','clus2','clus3'，规模为 1,2,3：
    sizes = {
        'm_F1_h3': 4,
        'm_F2_h3': 1,
        'm_F3_h3': 1,
        'm_F4_h3': 1,
    }

    # 计算 D_CCJS 距离矩阵
    div_df = divergence_matrix(bbas, sizes)
    met_df = metric_matrix(bbas, sizes)

    # 控制台输出距离矩阵
    print("\n----- D_CCJS Divergence 矩阵 -----")
    print(div_df.to_string())
    print("\n----- D_CCJS Metric 矩阵 -----")
    print(met_df.to_string())

    # 保存到 CSV（experiments_result 目录）
    save_csv(div_df, default_name=csv_name, label='divergence')
    print(f"结果 CSV: experiments_result/d_ccjs_divergence_{os.path.splitext(csv_name)[0]}.csv")

    # 验证度量性质：非负性，对称性，三角不等式
    labels = list(met_df.index)

    # 1. 测试对称性
    test_symmetry(met_df)

    # 2. 测试非负性
    test_nonnegativity(met_df)

    # 3. 测试三角不等式
    test_triangle_inequality(met_df)
