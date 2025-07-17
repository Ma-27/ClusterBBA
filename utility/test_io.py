# -*- coding: utf-8 -*-
"""I/O 工具函数简单测试脚本"""

import os
import sys

import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from utility.io import load_bbas, parse_focal_set, format_set  # type: ignore


# ------------------------------ 测试 parse_focal_set ------------------------------ #
def test_parse_and_format():
    samples = ['', '∅', '{}', 'A', '{A ∪ B}', 'C ∪ D']
    print("测试 parse_focal_set 与 format_set:")
    for s in samples:
        fs = parse_focal_set(s)
        formatted = format_set(fs)
        print(f"输入: '{s}' -> parse_focal_set: {fs}, format_set: '{formatted}'")


# ------------------------------ 主函数 ------------------------------ #
if __name__ == '__main__':
    # todo 默认示例文件，可以灵活修改
    default_name = 'Example_0.csv'
    # 处理命令行参数：CSV 文件名
    csv_name = sys.argv[1] if len(sys.argv) > 1 else default_name

    # 确定项目根目录：当前脚本位于 divergence/，故上溯一级
    base = os.path.dirname(os.path.abspath(__file__))
    # 构造数据文件路径（相对于项目根）
    csv_path = os.path.join(base, '..', 'data', 'examples', csv_name)
    if not os.path.isfile(csv_path):
        print(f"找不到 CSV 文件: {csv_path}")
        sys.exit(1)

    # 读取 CSV 并解析 BBA 数据
    df = pd.read_csv(csv_path)
    bbas, focal_cols = load_bbas(df)  # load_bbas 返回 ([BBA, …], focal_cols)

    print(bbas)

    print("焦元列:", focal_cols)

    # 1) 先测试 parse_focal_set
    test_parse_and_format()
