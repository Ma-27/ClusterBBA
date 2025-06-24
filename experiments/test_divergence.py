# -*- coding: utf-8 -*-
"""
test_divergence.py

复现 bjs.py 主函数流程，使用 divergence_calculation 包外导入：
- 处理命令行参数，指定 CSV 文件名
- 加载并解析 BBA
- 计算 BJS 距离矩阵
- 控制台输出
- 保存 CSV
- 绘制并保存热力图

在项目根目录运行：
    python test_divergence.py [Example_3_3.csv]
"""

import os
import sys

import pandas as pd

from divergence_calculation.bjs import (
    load_bbas,
    distance_matrix,
    save_csv,
    plot_heatmap
)

if __name__ == '__main__':
    # todo 默认示例文件名
    default_name = 'Example_0_1.csv'
    # 支持通过命令行参数指定 CSV 文件名
    csv_name = sys.argv[1] if len(sys.argv) > 1 else default_name

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

    # 计算 BJS 距离矩阵
    dist_df = distance_matrix(bbas)

    # 控制台输出距离矩阵
    print("\n----- BJS 距离矩阵 -----")
    print(dist_df.to_string())

    # 保存到 CSV（experiments_result 目录）
    save_csv(dist_df, default_name=csv_name)
    print(f"结果 CSV: experiments_result/bjs_{os.path.splitext(csv_name)[0]}.csv")

    # 绘制并保存热力图
    plot_heatmap(dist_df, default_name=csv_name)
    print(f"可视化图已保存到: experiments_result/bjs_{os.path.splitext(csv_name)[0]}.png")
