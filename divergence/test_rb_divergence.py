# -*- coding: utf-8 -*-
"""test_rb_divergence.py

使用 ``rb_divergence`` 计算 RB 散度矩阵并验证度量性质。
在项目根目录运行：
    python test_rb_divergence.py [Example_3_3.csv]
"""

import os
import sys

import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from divergence.metric_test import test_nonnegativity, test_symmetry, test_triangle_inequality  # type: ignore
from divergence.rb_divergence import divergence_matrix, save_csv  # type: ignore
from utility.io import load_bbas  # type: ignore

if __name__ == '__main__':
    # todo 默认示例文件，可以灵活修改
    default_name = 'Example_3_2_5.csv'
    csv_name = sys.argv[1] if len(sys.argv) > 1 else default_name

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, '..', 'data', 'examples', csv_name)
    if not os.path.isfile(csv_path):
        print(f"找不到 CSV 文件: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    bbas, _ = load_bbas(df)

    div_df = divergence_matrix(bbas)

    print("\n----- RB Divergence 矩阵 -----")
    print(div_df.to_string())

    save_csv(div_df, default_name=csv_name, label='divergence')
    print(f"结果 CSV: experiments_result/rb_divergence_{os.path.splitext(csv_name)[0]}.csv")

    test_symmetry(div_df)
    test_nonnegativity(div_df)
    test_triangle_inequality(div_df)
