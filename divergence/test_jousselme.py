# -*- coding: utf-8 -*-
"""
测试 Jousselme 距离模块 test_jousselme.py
=========================================
复现 jousselme.py 主函数流程，使用 divergence 包外导入：
- 处理命令行参数，指定 CSV 文件名
- 加载并解析 BBA
- 计算 Jousselme 距离矩阵
- 控制台输出
- 保存 CSV
- 绘制并保存热力图

在项目根目录运行：
    python test_jousselme.py [Example_3_3.csv]
"""

import os
import sys

import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from divergence.jousselme import distance_matrix, save_csv  # type: ignore
from divergence.metric_test import test_nonnegativity, test_symmetry, test_triangle_inequality  # type: ignore
from utility.io import load_bbas

# ------------------------------ 主函数 ------------------------------ #
if __name__ == '__main__':
    # todo 默认示例文件，可以灵活修改
    default_name = 'Example_3_8.csv'

    # 处理命令行参数：CSV 文件名
    csv_name = sys.argv[1] if len(sys.argv) > 1 else default_name

    # 确定项目根目录：当前脚本位于 divergence/，故上溯一级
    base = os.path.dirname(os.path.abspath(__file__))
    # 构造数据文件路径（相对于项目根）
    csv_path = os.path.join(base, '..', 'data', 'examples', csv_name)
    if not os.path.isfile(csv_path):
        print(f"找不到 CSV 文件: {csv_path}")
        sys.exit(1)

    # 读取 CSV 并解析 BBA 数据
    df = pd.read_csv(csv_path)
    bbas, _ = load_bbas(df)

    # 计算 Jousselme 距离矩阵
    dist_df = distance_matrix(bbas)

    # ---------- 控制台输出 ---------- #
    print("\n----- Jousselme 距离矩阵 -----")
    print(dist_df.to_string())

    # 保存并可视化
    save_csv(dist_df, default_name=csv_name, label='distance')
    print(f"\nDistance 结果 CSV: experiments_result/jousselme_{'distance'}_{os.path.splitext(csv_name)[0]}.csv")

    # plot_heatmap(dist_df, default_name=csv_name, title='Jousselme Distance Heatmap', label='distance')
    # print(f"Distance 结果可视化: experiments_result/jousselme_{'distance'}_{os.path.splitext(csv_name)[0]}.png")

    # 验证度量性质：非负性，对称性，三角不等式
    # 1. 测试对称性
    test_symmetry(dist_df)

    # 2. 测试非负性
    test_nonnegativity(dist_df)

    # 3. 测试三角不等式
    test_triangle_inequality(dist_df)
