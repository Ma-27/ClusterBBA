# -*- coding: utf-8 -*-
"""
Deng Entropy Calculator 计算 BBA 的 Deng 熵
=====================
本脚本读取位于 data/examples 目录下的 CSV，对其中每一行 BBA 计算其 Deng 熵：

    E_d(m) = -∑_{A⊆X} m(A) · log2\left(\dfrac{m(A)}{2^{|A|}-1}\right)

其中空集和质量为 0 的焦元不参与计算。

输入输出格式、注释风格与 fractal_average.py 保持一致：
1. 从 CSV 读取数据，列名形如 "{A ∪ B}" 表示焦元。
2. 控制台打印计算结果，并将结果保存到 experiments_result 目录，
   文件名格式为 dengentropy_<原 CSV 文件名>.csv。
3. 所有数值保留四位小数。

使用方法
--------

在 experiments 目录下执行，例如：
$ python deng_entropy_calculator.py                # 默认 Example_3_3.csv
$ python deng_entropy_calculator.py Example_3_3_1.csv

"""

import math
import os
import sys
from typing import Dict, FrozenSet

import pandas as pd


# ------------------------------ 工具函数 ------------------------------ #

# 解析列名，去除花括号并按“∪”分割生成元素集，如 {A ∪ B} -> frozenset({'A', 'B'})
def parse_focal_set(cell: str) -> FrozenSet[str]:
    if cell.startswith("{") and cell.endswith("}"):
        cell = cell[1:-1]
    items = [e.strip() for e in cell.split("∪") if e.strip()]
    return frozenset(items)


# 将 FrozenSet 集合格式化为 BBA 字符串，如 frozenset({'A', 'B'}) -> {A ∪ B}
def format_set(s: FrozenSet[str]) -> str:
    if not s:
        return "∅"
    return "{" + " ∪ ".join(sorted(s)) + "}"


# ------------------------------ Deng 熵计算 ------------------------------ #

# 计算单个 BBA 的 Deng 熵
def deng_entropy(bba: Dict[FrozenSet[str], float]) -> float:
    entropy = 0.0
    for focal, mass in bba.items():
        # 跳过空集或质量为 0 的焦元
        if mass == 0 or len(focal) == 0:
            continue
        denom = (2 ** len(focal)) - 1
        # 理论上 denom>=1，此处无须额外判断
        entropy -= mass * math.log2(mass / denom)
    return entropy


# ------------------------------ I/O 与展示 ------------------------------ #

# 加载 CSV 并解析为 (名称, bba_dict) 列表
def load_bbas(df: pd.DataFrame):
    focal_cols = [c for c in df.columns if c != "BBA"]
    bbas = []
    for _, row in df.iterrows():
        bba: Dict[FrozenSet[str], float] = {}
        for col in focal_cols:
            bba[parse_focal_set(col)] = float(row[col])
        bbas.append((str(row.get("BBA", "m")), bba))
    return bbas


# ------------------------------ 主函数 ------------------------------ #

if __name__ == "__main__":
    #  todo 读取 CSV 文件名 可按需修改
    path = "Example_0.csv"
    try:
        csv_filename = sys.argv[1] if len(sys.argv) > 1 else path
    except IndexError:
        csv_filename = path

    # 定位 CSV 数据文件
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(base_dir, "..", "data", "examples")
    csv_path = os.path.normpath(os.path.join(csv_dir, csv_filename))

    if not os.path.isfile(csv_path):
        print(f"找不到 CSV 文件: {csv_path}")
        sys.exit(1)

    # 载入数据
    df = pd.read_csv(csv_path)
    bbas = load_bbas(df)

    # 计算并打印 Deng 熵
    results = []
    print("----- Deng Entropy 结果 -----")
    for name, bba in bbas:
        ed = deng_entropy(bba)
        print(f"{name}: {ed:.4f}")
        results.append([name, round(ed, 4)])

    # 保存结果到 experiments_result 目录
    result_dir = os.path.normpath(os.path.join(base_dir, "..", "experiments_result"))
    os.makedirs(result_dir, exist_ok=True)
    result_file = f"dengentropy_{os.path.splitext(csv_filename)[0]}.csv"
    result_path = os.path.join(result_dir, result_file)

    out_df = pd.DataFrame(results, columns=["BBA", "DengEntropy"])
    out_df.to_csv(result_path, index=False, float_format="%.4f")

    print(f"\n结果已保存到: {result_path}")
