# -*- coding: utf-8 -*-
"""
Average Fractal BBA Calculator
==============================

在 experiments_result 目录中，针对指定的单个 BBA 文件，
对齐焦元后计算该文件中所有已分形行的质量（Mass）平均值，并在控制台输出结果。
可通过修改 `target_file` 变量来指定需要平均的 CSV 文件名。

注意：只可用作计算 BBA 平均值！

"""
import os
import sys
from typing import Dict, FrozenSet, List

import pandas as pd


# ------------------------------ 工具函数 ------------------------------ #

def parse_focal_set(cell: str) -> FrozenSet[str]:
    """将列名字符串如 '{A ∪ B}' 解析为 frozenset({'A','B'})"""
    if cell.startswith('{') and cell.endswith('}'):
        inner = cell[1:-1]
        # 分割集合元素
        return frozenset(x.strip() for x in inner.split('∪'))
    return frozenset()


def format_set(s: FrozenSet[str]) -> str:
    """将 frozenset 转回格式化字符串，如 {'A','B'} -> '{A ∪ B}'"""
    if not s:
        return '∅'
    # 按字母排序并用 ' ∪ ' 连接
    return '{' + ' ∪ '.join(sorted(s)) + '}'


def load_bba_rows(path: str) -> (List[Dict[FrozenSet[str], float]], List[str]):
    """从 CSV 中加载所有 BBA 行，返回 BBA 字典列表与焦元列顺序"""
    df = pd.read_csv(path)
    # 假设第一列为标签，后续列为焦元质量值
    focal_cols = list(df.columns)[1:]
    bba_list: List[Dict[FrozenSet[str], float]] = []
    # 遍历每一行，提取质量值
    for _, row in df.iterrows():
        bba: Dict[FrozenSet[str], float] = {}
        for col in focal_cols:
            fs = parse_focal_set(col)
            bba[fs] = float(row[col])  # 将质量值转换为浮点数
        bba_list.append(bba)
    return bba_list, focal_cols


def bba_to_series(bba: Dict[FrozenSet[str], float], order: List[str]) -> List[float]:
    """按给定的原始列名顺序生成质量值列表，用于最后输出"""
    data = {format_set(fs): mass for fs, mass in bba.items()}
    # 对齐顺序，不存在则填 0.0
    return [data.get(col, 0.0) for col in order]


# ------------------------------ 主函数 ------------------------------ #
if __name__ == '__main__':
    # 用户在此处指定需要计算平均的 CSV 文件名
    target_file = 'fractal_Example_3_3_3_h0.csv'  # todo: 替换为所需文件名

    # 构造文件路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.normpath(os.path.join(base_dir, '..', 'experiments_result'))
    path = os.path.join(result_dir, target_file)
    if not os.path.isfile(path):
        print(f"未找到文件: {path}")
        sys.exit(1)

    # 加载 CSV 中的所有 BBA 行
    bba_list, focal_order = load_bba_rows(path)
    if not bba_list:
        print("CSV 中未包含任何 BBA 行。")
        sys.exit(1)

    # 累加各焦元质量
    mass_sum: Dict[FrozenSet[str], float] = {}
    for bba in bba_list:
        for fs, m in bba.items():
            # 初始化或累加
            mass_sum[fs] = mass_sum.get(fs, 0.0) + m
    # 行数即样本数，用于平均
    count = len(bba_list)

    # 计算平均质量
    avg_bba = {fs: mass / count for fs, mass in mass_sum.items()}

    # 准备输出表头与数据
    header = ['Label'] + focal_order
    row = ['avg'] + bba_to_series(avg_bba, focal_order)
    # 保留4位小数
    df_avg = pd.DataFrame([row], columns=header).round(4)

    # 打印结果
    print('----- 指定 CSV BBA 平均 -----')  # 分隔提示
    print(df_avg.to_string(index=False))  # 不显示索引
