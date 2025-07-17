# -*- coding: utf-8 -*-
"""test_ds_rules.py

演示 :mod:`fusion.ds_rule` 的基本用法：
 - 处理命令行参数指定 CSV 文件名
 - 加载并解析 BBA
 - 按 DS 组合规则逐步合并 BBA
 - 打印每一步的合并结果并保存最终结果

在项目根目录运行：
    python test_ds_rules.py [Example_3_2.csv]
"""

import os
import sys

import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from fusion.ds_rule import ds_combine
from utility.bba import BBA
from utility.io import load_bbas, save_bba

if __name__ == "__main__":
    # todo 默认示例文件名，可以根据需要修改
    default_name = "Example_3_2.csv"
    csv_name = sys.argv[1] if len(sys.argv) > 1 else default_name

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, "..", "data", "examples", csv_name)
    if not os.path.isfile(csv_path):
        print(f"找不到 CSV 文件: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    bbas, focal_cols = load_bbas(df)

    # 按顺序逐条合并，并展示每一步的结果
    step_results = []
    cur = None
    for idx, bba in enumerate(bbas, start=1):
        cur = bba if cur is None else ds_combine(cur, bba)
        step_results.append((idx, cur))

    print("\n----- 每一步 DS 组合结果 -----")
    for idx, bba in step_results:
        cols = [BBA.format_set(fs) for fs in sorted(bba.keys(),
                                                    key=BBA._set_sort_key)]
        step_df = pd.DataFrame([
            ["m"] + bba.to_series(cols)
        ], columns=["BBA"] + cols).round(4)

        print(f"\n[m1 ... m{idx}]")
        print(step_df.to_string(index=False))

    result = step_results[-1][1]

    cols = [BBA.format_set(fs) for fs in sorted(result.keys(),
                                                key=BBA._set_sort_key)]
    out_df = pd.DataFrame([
        ["m"] + result.to_series(cols)
    ], columns=["BBA"] + cols).round(4)

    print("\n----- 最终合并结果 -----")
    print(out_df.to_string(index=False))

    out_name = f"combined_ds_rules_{os.path.splitext(csv_name)[0]}.csv"
    result_dir = os.path.normpath(os.path.join(base, '..', 'experiments_result'))
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, out_name)

    result.name = 'm'
    save_bba(result, focal_cols=None, out_path=result_path,
             default_name=out_name, float_format='%.4f')
    print(f"\n结果已保存到: {result_path}")
