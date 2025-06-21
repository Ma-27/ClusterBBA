# -*- coding: utf-8 -*-
"""
BBA Fractal Operator 计算一组（或者一个BBA）的 seg_depth 阶分形
====================
实现对输入的基本概率分配（BBA）进行 **均等分形**：父焦元 A_j 的质量 m(A_j) 被平均分配到其所有非空子集 A_i 上，
每个子集获得的系数均为 1 / (2^{|A_j|} - 1)。当 |A_j| = 1 时，分母为 1，质量保持不变。
多阶分型依旧通过迭代一次分型实现，参数 h 仅表示迭代次数，本身不再出现在公式中。
定义 0 阶分形时直接返回原始 BBA，不支持负数分形。

在输出时保留四位小数。

使用方法
--------
在 data/examples 文件夹中放置了 CSV 文件，例如 Example_3_3.csv。

脚本可直接运行，无需命令行参数时默认使用 h=1（注意要转到目录 cd experiments）：
$ python fractal_average.py             # 等价于 h=1
或指定阶数 h：
$ python fractal_average.py 3          # h = 3

"""

import itertools
import os
import sys
from typing import Dict, FrozenSet, List

import pandas as pd


# ------------------------------ 工具函数 ------------------------------ #

# 解析列名，去除花括号并按“∪”分割生成元素集，如 {A ∪ B} -> frozenset({'A', 'B'})
def parse_focal_set(cell: str) -> FrozenSet[str]:
    if cell.startswith("{") and cell.endswith("}"):
        cell = cell[1:-1]
    items = [e.strip() for e in cell.split("∪") if e.strip()]
    return frozenset(items)


# 生成 s 的所有 *非空* 子集（包括自身）。
def powerset(s: FrozenSet[str]):
    items = list(s)
    for r in range(1, len(items) + 1):
        for combo in itertools.combinations(items, r):
            yield frozenset(combo)


# ------------------------------ 分形核心 ------------------------------ #

# 执行1阶分形，给各个子集均分质量。
def split_once(bba: Dict[FrozenSet[str], float], _h: int) -> Dict[FrozenSet[str], float]:
    new_bba: Dict[FrozenSet[str], float] = {}
    for Aj, mass in bba.items():
        # 单元素焦元或空集按原质量保留或均分到子集
        if len(Aj) == 0:
            # 空集质量不参与分型，直接保留
            new_bba[Aj] = new_bba.get(Aj, 0.0) + mass
            continue
        denom = (2 ** len(Aj)) - 1  # 非空子集数，此处为
        for Ai in powerset(Aj):
            factor = 1.0 / denom
            new_bba[Ai] = new_bba.get(Ai, 0.0) + mass * factor
    return new_bba


# 执行 h 阶均等分型（迭代 h 次 split_once）。
def higher_order_bba(bba: Dict[FrozenSet[str], float], h: int) -> Dict[FrozenSet[str], float]:
    current = bba
    for _ in range(h):
        current = split_once(current, h)
    return current


# ------------------------------ I/O 与展示 ------------------------------ #

# 从数据集中加载BBA
def load_bbas(df: pd.DataFrame):
    # 提取列名（跳过 'BBA' 列）作为焦元表示
    focal_cols = [c for c in df.columns if c != "BBA"]  # 焦元列名
    bbas = []  # List[Tuple[name, bba_dict]]
    # 将每行转换为 (名称, bba_dict) 元组列表
    for _, row in df.iterrows():
        bba: Dict[FrozenSet[str], float] = {}
        for col in focal_cols:
            # 将列名转焦元集合，行值转质量
            bba[parse_focal_set(col)] = float(row[col])
        bbas.append((str(row.get("BBA", "m")), bba))  # 存储名称与 BBA
    return bbas, focal_cols


# 将 FrozenSet 集合格式化为 BBA 字符串，如 frozenset({'A', 'B'}) -> {A ∪ B}
def format_set(s: FrozenSet[str]) -> str:
    if not s:
        return "∅"
    return "{" + " ∪ ".join(sorted(s)) + "}"


# 按照指定的焦元顺序，将 bba_dict 转为数值列表，用于构造 DataFrame 行
def bba_to_series(bba: Dict[FrozenSet[str], float], focal_order: List[str]):
    data = {format_set(k): v for k, v in bba.items()}
    return [data.get(col, 0.0) for col in focal_order]


# ------------------------------ 主函数 ------------------------------ #
if __name__ == '__main__':
    # 读取 h，允许 h=0，禁止负数
    try:
        # todo 分形深度 可按需修改
        seg_depth = 3
        h = int(sys.argv[1]) if len(sys.argv) > 1 else seg_depth
        if h < 0:
            raise ValueError
    except ValueError:
        print("参数 h 必须是不小于 0 的整数，示例：python fractal_average.py 2 或 python fractal_average.py")
        sys.exit(1)

    # 定位 CSV 数据
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(base_dir, '..', 'data', 'examples')
    # todo 分形的 BBA 数据位置，可按需修改
    csv_file = 'Example_3_3_2.csv'
    csv_path = os.path.normpath(os.path.join(csv_dir, csv_file))
    if not os.path.isfile(csv_path):
        print(f"找不到 CSV 文件: {csv_path}")
        sys.exit(1)

    # 载入 BBA
    df = pd.read_csv(csv_path)
    bbas, focal_cols = load_bbas(df)

    # 迭代 0~h 阶分型并打印结果
    out_k = None
    for k in range(0, h + 1):
        print(f"----- 分型结果 (h = {k}) -----")
        rows_k = []
        for name, bba in bbas:
            if k == 0:
                fbba_k = bba  # 0阶：原始 BBA
            else:
                # 对 BBA 做 k 阶分型
                fbba_k = higher_order_bba(bba, k)
            rows_k.append([f"{name}_h{k}"] + bba_to_series(fbba_k, focal_cols))
        out_k = pd.DataFrame(rows_k, columns=["BBA"] + focal_cols).round(4)
        # 打印第 k 轮结果
        print(out_k.to_string(index=False))

    # 保存最终结果到 experiments_result 目录
    result_dir = os.path.normpath(os.path.join(base_dir, '..', 'experiments_result'))
    os.makedirs(result_dir, exist_ok=True)
    # 构造结果文件路径，文件名包含原始 CSV 名称与使用的 h 阶
    result_file = f"fractal_{os.path.splitext(csv_file)[0]}_h{h}.csv"
    result_path = os.path.join(result_dir, result_file)
    # 将最后一阶结果保存为 CSV 文件，所有数值以四位小数格式输出
    out_k.to_csv(result_path, index=False, float_format='%.4f')
    print(f"\n结果已保存到: {result_path}")
