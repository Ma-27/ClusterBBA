# -*- coding: utf-8 -*-
"""Xiao BJS‑Belief Entropy 组合规则 — 数值示例验证"""

import os
import sys

import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from fusion.xiao_bjs_rule import (
    xiao_bjs_combine,
    _weighted_average_bba,
    origin_credibility_degrees,
    information_volume,
    credibility_degrees,
)
from utility.bba import BBA
from utility.io import load_bbas, save_bba


def _print_step(k: int, combined: BBA, crd: list[float], iv: list[float], weight: list[float]) -> None:
    """打印第 ``k`` 步的组合结果。"""
    cols = [BBA.format_set(fs) for fs in sorted(combined.keys(), key=BBA._set_sort_key)]
    table = pd.DataFrame([["m"] + combined.to_series(cols)], columns=["BBA"] + cols).round(4)
    crd_str = ", ".join(f"Crd{i + 1}={c:.4f}" for i, c in enumerate(crd))
    iv_str = ", ".join(f"ĨV{i + 1}={v:.4f}" for i, v in enumerate(iv))
    acrd_str = ", ".join(
        f"ÃCrd{i + 1}={w:.4f}" for i, w in enumerate(weight)
    )

    print(f"\n[m1 ... m{k}]")
    print(crd_str)
    print(iv_str)
    print(acrd_str)
    print(table.to_string(index=False))


def _print_average(k: int, avg: BBA) -> None:
    cols = [BBA.format_set(fs) for fs in sorted(avg.keys(), key=BBA._set_sort_key)]
    table = pd.DataFrame([["avg"] + avg.to_series(cols)], columns=["BBA"] + cols).round(4)
    print(f"\n[m1 ... m{k}] 加权平均 BBA")
    print(table.to_string(index=False))


if __name__ == "__main__":
    # todo 默认示例 CSV；可通过命令行参数替换
    default_csv = "Example_3_2.csv"
    csv_name = sys.argv[1] if len(sys.argv) > 1 else default_csv

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.normpath(os.path.join(base_dir, "..", "data", "examples", csv_name))
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")

    df = pd.read_csv(csv_path)
    raw_bbas, _ = load_bbas(df)

    print("\n----- Xiao BJS‑Belief Entropy 组合结果 -----")
    for k in range(2, len(raw_bbas) + 1):
        cur_bbas = [b for _, b in raw_bbas[:k]]
        crd = origin_credibility_degrees(cur_bbas)
        iv = information_volume(cur_bbas)
        weight = credibility_degrees(cur_bbas)

        # todo 注意以下两个指标的给定，大部分情况下是不给定的，但是如果给定则需要取消后面一连串的注释。
        # 充分度指标（静态可选）
        # mu = [1.0, 0.60, 1.0][:k]
        # 重要度指标（静态可选）
        # nu = [1.0, 0.90, 0.60][:k]

        avg_bba = _weighted_average_bba(cur_bbas)
        # avg_bba = _weighted_average_bba(cur_bbas, mu=mu, nu=nu)
        # _print_average(k, avg_bba)  # 若需展开平均 BBA，可取消注释

        combined_bba = xiao_bjs_combine(cur_bbas)
        # combined_bba = xiao_bjs_combine(cur_bbas, mu=mu, nu=nu)
        _print_step(k, combined_bba, crd, iv, weight)

    out_name = f"combined_xiao_bjs_{os.path.splitext(csv_name)[0]}.csv"
    result_dir = os.path.normpath(os.path.join(base_dir, "..", "experiments_result"))
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, out_name)
    save_bba(combined_bba, name="m", focal_cols=None, out_path=result_path, default_name=out_name, float_format="%.4f")
    print(f"\n结果已保存到: {result_path}")
