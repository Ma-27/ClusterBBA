# -*- coding: utf-8 -*-
"""
BBA Fractal Operator
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

def parse_focal_set(cell: str) -> FrozenSet[str]:
    """{A ∪ B} -> frozenset({'A', 'B'})"""  # 解析列名，去除花括号并按“∪”分割生成元素集
    if cell.startswith("{") and cell.endswith("}"):
        cell = cell[1:-1]
    items = [e.strip() for e in cell.split("∪") if e.strip()]
    return frozenset(items)


def powerset(s: FrozenSet[str]):
    """生成 s 的所有 *非空* 子集（包括自身）。"""
    items = list(s)
    for r in range(1, len(items) + 1):
        for combo in itertools.combinations(items, r):
            yield frozenset(combo)


# ------------------------------ 分型核心 ------------------------------ #

def split_once(bba: Dict[FrozenSet[str], float], _h: int) -> Dict[FrozenSet[str], float]:
    new_bba: Dict[FrozenSet[str], float] = {}
    for Aj, mass in bba.items():
        if len(Aj) == 0:  # 空集质量保持不变
            new_bba[Aj] = new_bba.get(Aj, 0.0) + mass
            continue
        denom = (2 ** len(Aj)) - 1  # 非空子集数量
        for Ai in powerset(Aj):
            factor = 1.0 / denom
            new_bba[Ai] = new_bba.get(Ai, 0.0) + mass * factor
    return new_bba


def higher_order_bba(bba: Dict[FrozenSet[str], float], h: int) -> Dict[FrozenSet[str], float]:
    """执行 h 阶均等分型（迭代 h 次 split_once）。"""
    current = bba
    for _ in range(h):
        current = split_once(current, h)
    return current


# ------------------------------ I/O 与展示 ------------------------------ #

def load_bbas(df: pd.DataFrame):
    focal_cols = [c for c in df.columns if c != "BBA"]
    bbas = []  # List[Tuple[name, bba_dict]]
    for _, row in df.iterrows():
        bba: Dict[FrozenSet[str], float] = {}
        for col in focal_cols:
            bba[parse_focal_set(col)] = float(row[col])
        bbas.append((str(row.get("BBA", "m")), bba))
    return bbas, focal_cols


def format_set(s: FrozenSet[str]) -> str:
    """将集合格式化回字符串，如 {'A','B'} -> {A ∪ B}"""
    if not s:
        return "∅"
    return "{" + " ∪ ".join(sorted(s)) + "}"


def bba_to_series(bba: Dict[FrozenSet[str], float], focal_order: List[str]):
    # 按列顺序生成数值列表，便于 DataFrame 输出
    data = {format_set(k): v for k, v in bba.items()}
    return [data.get(col, 0.0) for col in focal_order]


# ------------------------------ 主函数 ------------------------------ #
if __name__ == '__main__':
    # 读取 h，允许 h=0，禁止负数
    try:
        h = int(sys.argv[1]) if len(sys.argv) > 1 else 1  # fixme h 可按需修改
        if h < 0:
            raise ValueError
    except ValueError:
        print("参数 h 必须是不小于 0 的整数，示例：python fractal_average.py 2 或 python fractal_average.py")
        sys.exit(1)

    # 定位 CSV 文件目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(base_dir, '..', 'data', 'examples')
    # fixme 加载数据集，可按需修改
    csv_file = 'Example_3_3_2.csv'
    csv_path = os.path.normpath(os.path.join(csv_dir, csv_file))
    if not os.path.isfile(csv_path):
        print(f"找不到 CSV 文件: {csv_path}")
        sys.exit(1)

    # 读取数据并生成 BBA 列表
    df = pd.read_csv(csv_path)
    bbas, focal_cols = load_bbas(df)

    # 遍历 0~h 阶分型，打印结果，并保存最后一阶
    out_k = None
    for k in range(0, h + 1):
        print(f"----- 分型结果 (h = {k}) -----")
        rows_k = []
        for name, bba in bbas:
            if k == 0:
                fbba_k = bba  # 0 阶直接原样
            else:
                fbba_k = higher_order_bba(bba, k)
            rows_k.append([f"{name}_h{k}"] + bba_to_series(fbba_k, focal_cols))
        out_k = pd.DataFrame(rows_k, columns=["BBA"] + focal_cols).round(4)
        # 打印第 k 轮结果
        print(out_k.to_string(index=False))

    # ------------------------------ 保存结果 ------------------------------ #
    # 创建输出结果文件夹 experiments_result，如果不存在则自动生成
    result_dir = os.path.normpath(os.path.join(base_dir, '..', 'experiments_result'))
    os.makedirs(result_dir, exist_ok=True)
    # 构造结果文件路径，文件名包含原始 CSV 名称与使用的 h 阶
    result_file = f"fractal_{os.path.splitext(csv_file)[0]}_h{h}.csv"
    result_path = os.path.join(result_dir, result_file)
    # 将最后一阶结果保存为 CSV 文件，所有数值以四位小数格式输出
    out_k.to_csv(result_path, index=False, float_format='%.4f')
    print(f"\n结果已保存到: {result_path}")
    