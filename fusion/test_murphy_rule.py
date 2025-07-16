# -*- coding: utf-8 -*-
"""Murphy 平均组合规则——数值示例验证"""

import os
import sys

import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from fusion.murphy_rule import (
    murphy_combine,
    _average_bba,
)
from utility.bba import BBA
from utility.io import load_bbas, save_bba


def _print_step(k: int, combined: BBA) -> None:
    """打印第 k 步组合结果（m₁⋯m_k）"""
    cols = [BBA.format_set(fs)
            for fs in sorted(combined.keys(), key=BBA._set_sort_key)]
    table = pd.DataFrame([["m"] + combined.to_series(cols)],
                         columns=["BBA"] + cols).round(4)
    print(f"\n[m1 ... m{k}]")
    print(table.to_string(index=False))


def _print_average(k: int, avg: BBA) -> None:
    """打印第 k 步的平均 BBA（参与 n − 1 次合并前）"""
    cols = [BBA.format_set(fs)
            for fs in sorted(avg.keys(), key=BBA._set_sort_key)]
    table = pd.DataFrame([["avg"] + avg.to_series(cols)],
                         columns=["BBA"] + cols).round(4)
    print(f"\n[m1 ... m{k}] 平均 BBA")
    print(table.to_string(index=False))


if __name__ == "__main__":
    # todo 默认示例文件，可通过命令行参数替换
    default_csv = "Example_3_2.csv"
    csv_name = sys.argv[1] if len(sys.argv) > 1 else default_csv

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.normpath(os.path.join(base_dir, "..", "data", "examples", csv_name))
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")

    df = pd.read_csv(csv_path)
    raw_bbas, _ = load_bbas(df)  # [(name, BBA), …]

    print("\n----- Murphy 组合结果 -----")
    for k in range(2, len(raw_bbas) + 1):
        cur_bbas = [b for _, b in raw_bbas[:k]]

        # 打印平均BBA，仅封装了一下。
        avg_bba = _average_bba(cur_bbas)
        # _print_average(k, avg_bba)

        # 使用murphy规则去组合证据。
        combined_bba = murphy_combine(cur_bbas)
        _print_step(k, combined_bba)

    # 保存最后一次（全部证据）组合结果
    out_name = f"combined_murphy_{os.path.splitext(csv_name)[0]}.csv"
    result_dir = os.path.normpath(os.path.join(base_dir, "..", "experiments_result"))
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, out_name)
    save_bba(combined_bba, name="m", focal_cols=None,
             out_path=result_path, default_name=out_name, float_format="%.4f")
    print(f"\n结果已保存到: {result_path}")
