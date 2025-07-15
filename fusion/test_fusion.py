# -*- coding: utf-8 -*-
"""test_fusion.py

多种融合规则对比测试脚本
=========================

模仿 ``test_xiao_rb_rule.py`` 的风格，逐步融合 CSV 中的 BBA，在每一趟融合结束后打印两张表：
1. 各方法得到的融合 BBA；
2. 每条 BBA 在各方法下的权重。

脚本结束时，会使用 Markdown 表格再打印一次最终融合结果与权重。

用法::

    python test_fusion.py [csv_filename]

若未指定文件名，默认读取 ``Example_3_2.csv``。
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

# 表格输出时启用中文对齐
pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.unicode.ambiguous_as_wide", True)

# 依赖本项目内现成工具函数 / 模块
from config import PROGRESS_NCOLS
from fusion.ds_rule import combine_multiple
from fusion.murphy_rule import murphy_combine, credibility_degrees as murphy_weights
from fusion.deng_mae_rule import (
    modified_average_evidence,
    credibility_degrees as deng_weights,
)
from fusion.xiao_bjs_pure_rule import (
    xiao_bjs_pure_combine,
    credibility_degrees as bjs_pure_weights,
)
from fusion.xiao_bjs_rule import (
    xiao_bjs_combine,
    credibility_degrees as bjs_weights,
)
from fusion.xiao_rb_rule import (
    xiao_rb_combine,
    credibility_degrees as rb_weights,
)
from fusion.my_rule import (
    my_combine,
    credibility_degrees as my_weights,
)
from utility.bba import BBA
from utility.io import load_bbas

# 需要比较的融合规则及其对应的组合函数
METHODS = [
    ("Dempster", combine_multiple),
    ("Murphy", murphy_combine),
    ("Deng", modified_average_evidence),
    ("BJS Pure (Xiao)", xiao_bjs_pure_combine),
    ("BJS Origin (Xiao)", xiao_bjs_combine),
    ("RB (Xiao)", xiao_rb_combine),
    ("Proposed", my_combine),
]


def collect_weights(bbas: List[BBA], names: List[str]) -> Dict[str, List[str]]:
    """按照各方法计算权重, 返回字符串列表用于展示。"""
    k = len(bbas)
    # Dempster 组合规则没有权重概念，用 '—' 填充
    weight_table: Dict[str, List[str]] = {
        "Dempster": ["—"] * k,
        "Murphy": [f"{w:.4f}" for w in murphy_weights(bbas)],
        "Deng": [f"{w:.4f}" for w in deng_weights(bbas)],
        "BJS Pure (Xiao)": [f"{w:.4f}" for w in bjs_pure_weights(bbas)],
        "BJS Origin (Xiao)": [f"{w:.4f}" for w in bjs_weights(bbas)],
        "RB (Xiao)": [f"{w:.4f}" for w in rb_weights(bbas)],
        "Proposed": [f"{w:.4f}" for w in my_weights(bbas, names)],
    }
    return weight_table


def print_tables(k: int, results: Dict[str, BBA], weights: Dict[str, List[str]],
                 names: List[str]) -> None:
    """打印第 ``k`` 步的融合结果表和权重表。"""
    all_sets = set()
    for bba in results.values():
        # 收集出现过的所有焦元，保证列顺序一致
        all_sets.update(bba.keys())
    cols = [BBA.format_set(fs) for fs in sorted(all_sets, key=BBA._set_sort_key)]

    rows = []
    for name, bba in results.items():
        # 不打印接近 0 的值，保持表格整洁
        series = ["" if abs(v) < 1e-12 else f"{v:.4f}" for v in bba.to_series(cols)]
        rows.append([name] + series)
    df_res = pd.DataFrame(rows, columns=["methods"] + cols)

    print(f"\n[m1 ... m{k}] 融合 BBA")
    print(tabulate(df_res, headers="keys", tablefmt="pipe", showindex=False))

    # 构造权重表
    weight_rows = []
    method_names = list(results.keys())
    for idx in range(k):
        row = [names[idx]]  # 使用实际的 BBA 名称
        for name in method_names:
            row.append(weights[name][idx])
        weight_rows.append(row)
    df_w = pd.DataFrame(weight_rows, columns=["BBA"] + method_names)
    print("\n权重")
    print(tabulate(df_w, headers="keys", tablefmt="pipe", showindex=False))


if __name__ == "__main__":
    # todo 默认示例文件，可通过命令行参数替换
    default_csv = "Example_3_2_3.csv"
    csv_name = sys.argv[1] if len(sys.argv) > 1 else default_csv

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.normpath(os.path.join(base_dir, "..", "data", "examples", csv_name))
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")

    df = pd.read_csv(csv_path)
    raw_bbas, _ = load_bbas(df)

    # 逐步引入新的 BBA 进行融合
    pbar = tqdm(range(2, len(raw_bbas) + 1), desc="融合进度", ncols=PROGRESS_NCOLS)
    final_results = None
    final_weights = None
    for k in pbar:
        names = [name for name, _ in raw_bbas[:k]]
        # 当前参与融合的 BBA 集合
        cur_bbas = [b for _, b in raw_bbas[:k]]
        results: Dict[str, BBA] = {}
        # 不同规则分别计算融合结果
        for name, func in METHODS:
            if func is my_combine:
                results[name] = func(cur_bbas, names)
            else:
                results[name] = func(cur_bbas)
        weights = collect_weights(cur_bbas, names)
        # 打印当前轮的表格
        print_tables(k, results, weights, names)
        # 记录最后一轮结果，稍后保存
        final_results = results
        final_weights = weights

    if final_results is not None and final_weights is not None:
        # 整理所有出现过的焦元
        all_sets = set()
        for bba in final_results.values():
            all_sets.update(bba.keys())
        cols = [BBA.format_set(fs) for fs in sorted(all_sets, key=BBA._set_sort_key)]
        rows = []
        for name, bba in final_results.items():
            rows.append([name] + bba.to_series(cols))
        df_res = pd.DataFrame(rows, columns=["methods"] + cols)

        weight_rows = []
        method_names = list(final_results.keys())
        for idx, (bba_name, _) in enumerate(raw_bbas):
            row = [bba_name]
            for name in method_names:
                row.append(final_weights[name][idx])
            weight_rows.append(row)
        df_w = pd.DataFrame(weight_rows, columns=["BBA"] + method_names)

        # 结果保存目录和文件名
        out_base = os.path.splitext(csv_name)[0]
        result_dir = os.path.normpath(os.path.join(base_dir, "..", "experiments_result"))
        os.makedirs(result_dir, exist_ok=True)
        res_path = os.path.join(result_dir, f"fusion_results_{out_base}.csv")
        w_path = os.path.join(result_dir, f"fusion_weights_{out_base}.csv")
        df_res.to_csv(res_path, index=False, float_format="%.4f")
        df_w.to_csv(w_path, index=False)
        print(f"\n融合结果已保存到: {res_path}")
        print(f"权重信息已保存到: {w_path}")

        print("\n----- 最终融合结果 -----")
        print(tabulate(df_res, headers="keys", tablefmt="pipe", showindex=False))
        print("\n----- 最终权重 -----")
        print(tabulate(df_w, headers="keys", tablefmt="pipe", showindex=False))
