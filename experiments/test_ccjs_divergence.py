# -*- coding: utf-8 -*-
"""
test_ccjs_divergence.py

使用 divergence_calculation.ccjs 包计算 CCJS 距离矩阵，支持传入超参数 n，并验证度量性质
"""

import os
import sys

import pandas as pd

from divergence_calculation.ccjs import (
    load_bbas,
    metric_matrix,
    save_csv
)

if __name__ == '__main__':
    # todo 默认示例文件名，可根据实际情况修改
    default_name = 'Example_0_2.csv'
    # 支持通过命令行参数指定 CSV 文件名
    csv_name = sys.argv[1] if len(sys.argv) > 1 else default_name
    # 支持通过命令行参数指定超参数 n，默认为 1
    try:
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    except ValueError:
        print(f"超参数 n 无效: {sys.argv[2]}，请提供正整数！")
        sys.exit(1)

    # 确定项目根目录：当前脚本位于 divergence_calculation/，故上溯一级
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 构造数据文件路径（相对于项目根）
    csv_path = os.path.join(project_root, 'data', 'examples', csv_name)
    if not os.path.isfile(csv_path):
        print(f"找不到 CSV 文件: {csv_path}")
        sys.exit(1)

    # 读取 CSV 并解析 BBA 数据
    df = pd.read_csv(csv_path, encoding='utf-8')
    bbas, _ = load_bbas(df)

    # todo 构造簇规模参数：请手动指定各簇的规模（根据实际 bbas 名称修改）
    # 例如，若有三个簇，名称分别为 'clus1','clus2','clus3'，规模为 1,2,3：
    sizes = {
        'm_F1_h3': 4,
        'm_F2_h3': 1,
        'm_F3_h3': 1,
        'm_F4_h3': 1,
    }

    # 计算 CCJS 距离矩阵
    met_df = metric_matrix(bbas, sizes)

    # 控制台输出距离矩阵
    print("\n----- CCJS 距离矩阵 -----")
    print(met_df.to_string())

    # 保存到 CSV（experiments_result 目录）
    save_csv(met_df, default_name=csv_name)
    print(f"结果 CSV: experiments_result/ccjs_{"metric"}_{os.path.splitext(csv_name)[0]}.csv")

    # 绘制并保存热力图
    # plot_heatmap(met_df, default_name=csv_name)
    # print(f"可视化图已保存到: experiments_result/ccjs_{"metric"}_{os.path.splitext(csv_name)[0]}.png")

