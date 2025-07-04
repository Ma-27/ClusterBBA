# -*- coding: utf-8 -*-
r"""
Deng Entropy Calculator 计算 BBA 的 Deng 熵
=====================
本脚本读取位于 data/examples 目录下的 CSV，对其中每一行 BBA 计算其 Deng 熵：

    E_d(m) = -∑_{A⊆X} m(A) · log2\left(\dfrac{m(A)}{2^{|A|}-1}\right)

其中空集和质量为 0 的焦元不参与计算。

Deng 熵是衡量 BBA 不确定性的一种度量，越大表示不确定性越高。本脚本可供外部脚本调用计算 Deng 熵。

模块接口
--------
- deng_entropy(bba: BBA) -> float

使用方法
--------

在 experiments 目录下执行，例如：
$ python deng_entropy.py                # 默认 Example_3_3.csv
$ python deng_entropy.py Example_3_3_1.csv

也可以在外部脚本中导入以上模块接口。
"""

import math
import os
import sys

import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from utility.bba import BBA
from utility.io import load_bbas

__all__ = [
    'deng_entropy',
]


# ------------------------------ Deng 熵计算 ------------------------------ #

# 计算单个 BBA 的 Deng 熵
def deng_entropy(bba: BBA) -> float:
    entropy = 0.0
    for focal, mass in bba.items():
        # 跳过空集或质量为 0 的焦元
        if mass == 0 or len(focal) == 0:
            continue
        denom = (2 ** len(focal)) - 1
        # 理论上 denom>=1，此处无须额外判断
        entropy -= mass * math.log2(mass / denom)
    return entropy


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
    bbas, focal_cols = load_bbas(df)  # load_bbas 返回 ([(name, bba_dict), …], focal_cols)

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
