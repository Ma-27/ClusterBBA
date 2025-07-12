# -*- coding: utf-8 -*-
"""Modified Average Combination Rule — 数值示例验证"""

import os
import sys

import pandas as pd

from fusion.deng_mae_rule import (
    modified_average_evidence,
    _weighted_average_bba,
    credibility_degrees,
)
from utility.bba import BBA
from utility.io import load_bbas, save_bba


def _print_step(k: int, combined: BBA, crd: list[float]) -> None:
    cols = [BBA.format_set(fs) for fs in sorted(combined.keys(), key=BBA._set_sort_key)]
    table = pd.DataFrame([["m"] + combined.to_series(cols)],
                         columns=["BBA"] + cols).round(4)
    cred_str = ", ".join(f"C{i + 1}={c:.4f}" for i, c in enumerate(crd))
    print(f"\n[m1 … m{k}]  Credibility: {cred_str}")
    print(table.to_string(index=False))


def _print_average(k: int, avg: BBA) -> None:
    cols = [BBA.format_set(fs) for fs in sorted(avg.keys(), key=BBA._set_sort_key)]
    table = pd.DataFrame([['avg'] + avg.to_series(cols)],
                         columns=['BBA'] + cols).round(4)
    print(f"\n[m1 … m{k}] 加权平均 BBA")
    print(table.to_string(index=False))


if __name__ == "__main__":
    # todo 默认示例文件名，可以根据需要修改
    default_csv = "Example_3_2.csv"
    csv_name = sys.argv[1] if len(sys.argv) > 1 else default_csv

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.normpath(os.path.join(base_dir, "..", "data", "examples", csv_name))
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")

    df = pd.read_csv(csv_path)
    raw_bbas, _ = load_bbas(df)

    print("\n----- Modified Average Evidence组合结果 -----")
    for k in range(2, len(raw_bbas) + 1):
        cur_bbas = [b for _, b in raw_bbas[:k]]

        # 计算每条 BBA 的可信度权重，d 为 Jousselme Distance
        crd = credibility_degrees(cur_bbas)

        avg_bba = _weighted_average_bba(cur_bbas)
        # _print_average(k, avg_bba)

        combined_bba = modified_average_evidence(cur_bbas)
        _print_step(k, combined_bba, crd)

    out_name = f"combined_modified_avg_evidence_{os.path.splitext(csv_name)[0]}.csv"
    result_dir = os.path.normpath(os.path.join(base_dir, "..", "experiments_result"))
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, out_name)
    save_bba(combined_bba, name="m", focal_cols=None,
             out_path=result_path, default_name=out_name, float_format="%.4f")
    print(f"\n结果已保存到: {result_path}")
