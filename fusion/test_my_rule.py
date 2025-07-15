# -*- coding: utf-8 -*-
"""test_my_rule.py

数值示例验证 — Proposed Fusion Rule
=====================================

在命令行传入示例 CSV 文件名（默认 ``Example_3_2.csv``），逐步组合 BBA，
打印每一步的权重和融合结果，并将最终融合结果保存至 ``experiments_result``
目录下。
"""

from __future__ import annotations

import os
import sys
from typing import List

import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from fusion.my_rule import (
    my_combine,
    _weighted_average_bba,
    credibility_degrees,
)
from utility.bba import BBA
from utility.io import load_bbas, save_bba


def _print_step(k: int, combined: BBA, crd: List[float]) -> None:
    """打印第 ``k`` 步组合结果及每条证据的权重。"""
    cols = [BBA.format_set(fs) for fs in sorted(combined.keys(), key=BBA._set_sort_key)]
    table = pd.DataFrame([["m"] + combined.to_series(cols)], columns=["BBA"] + cols).round(4)
    crd_str = ", ".join(f"Crd{i + 1}={c:.4f}" for i, c in enumerate(crd))
    print(f"\n[m1 ... m{k}]")
    print(crd_str)
    print(table.to_string(index=False))


def _print_average(k: int, avg: BBA) -> None:
    """辅助打印加权平均 BBA，调试时可启用。"""
    cols = [BBA.format_set(fs) for fs in sorted(avg.keys(), key=BBA._set_sort_key)]
    table = pd.DataFrame([["avg"] + avg.to_series(cols)], columns=["BBA"] + cols).round(4)
    print(f"\n[m1 ... m{k}] 加权平均 BBA")
    print(table.to_string(index=False))


if __name__ == "__main__":
    # todo 默认示例 CSV，可在命令行指定其他文件
    default_csv = "Example_3_2.csv"
    csv_name = sys.argv[1] if len(sys.argv) > 1 else default_csv

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.normpath(os.path.join(base_dir, "..", "data", "examples", csv_name))
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")

    # 读取并解析 CSV 中的 BBA
    df = pd.read_csv(csv_path)
    raw_bbas, _ = load_bbas(df)

    print("\n----- Proposed Fusion Rule 组合结果 -----")
    for k in range(2, len(raw_bbas) + 1):
        cur_bbas = [b for _, b in raw_bbas[:k]]
        crd = credibility_degrees(cur_bbas)

        avg_bba = _weighted_average_bba(cur_bbas)
        # _print_average(k, avg_bba)

        combined_bba = my_combine(cur_bbas)
        _print_step(k, combined_bba, crd)

    out_name = f"combined_my_rule_{os.path.splitext(csv_name)[0]}.csv"
    result_dir = os.path.normpath(os.path.join(base_dir, "..", "experiments_result"))
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, out_name)
    # 保存最终融合结果
    save_bba(combined_bba, name="m", focal_cols=None, out_path=result_path,
             default_name=out_name, float_format="%.4f")
    print(f"\n结果已保存到: {result_path}")
