# -*- coding: utf-8 -*-
"""test_xiao_rb_rule.py

数值示例验证 — Xiao RB 组合规则
===================================

用法：
    python test_xiao_rb_rule.py [csv_filename]

若未指定 CSV，默认读取 ``Example_3_8.csv``（与论文同源的数据文件）。

CSV 约定：
    第一列为标签 ``BBA``，其余列为焦元质量，列名形如 ``"{A}"``。
"""
from __future__ import annotations

import os
import sys
from typing import List

import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from fusion.xiao_rb_rule import (
    xiao_rb_combine,
    _weighted_average_bba,
    credibility_degrees,
)
from utility.bba import BBA
from utility.io import load_bbas, save_bba


def _print_step(k: int, combined: BBA, crd: List[float]) -> None:
    cols = [BBA.format_set(fs) for fs in sorted(combined.keys(), key=BBA._set_sort_key)]
    table = pd.DataFrame([
        ["m"] + combined.to_series(cols)
    ], columns=["BBA"] + cols).round(4)
    crd_str = ", ".join(f"Crd{i + 1}={c:.4f}" for i, c in enumerate(crd))
    print(f"\n[m1 ... m{k}]")
    print(crd_str)
    print(table.to_string(index=False))


def _print_average(k: int, avg: BBA):
    cols = [BBA.format_set(fs) for fs in sorted(avg.keys(), key=BBA._set_sort_key)]
    table = pd.DataFrame([
        ["avg"] + avg.to_series(cols)
    ], columns=["BBA"] + cols).round(4)
    print(f"\n[m1 ... m{k}] 加权平均 BBA")
    print(table.to_string(index=False))


if __name__ == "__main__":
    # todo 默认示例 CSV；灵活实验替换
    default_csv = "Example_3_2.csv"
    csv_name = sys.argv[1] if len(sys.argv) > 1 else default_csv

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.normpath(os.path.join(base_dir, "..", "data", "examples", csv_name))
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")

    df = pd.read_csv(csv_path)
    raw_bbas, _ = load_bbas(df)

    print("\n----- Xiao RB 组合结果 -----")
    for k in range(2, len(raw_bbas) + 1):
        cur_bbas = raw_bbas[:k]
        crd = credibility_degrees(cur_bbas)

        avg_bba = _weighted_average_bba(cur_bbas)
        # _print_average(k, avg_bba)  # 若需展开平均 BBA，可取消注释

        combined_bba = xiao_rb_combine(cur_bbas)
        _print_step(k, combined_bba, crd)

    out_name = f"combined_xiao_rb_{os.path.splitext(csv_name)[0]}.csv"
    result_dir = os.path.normpath(os.path.join(base_dir, "..", "experiments_result"))
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, out_name)
    combined_bba.name = "m"
    save_bba(combined_bba, focal_cols=None,
             out_path=result_path, default_name=out_name, float_format="%.4f")
    print(f"\n结果已保存到: {result_path}")
