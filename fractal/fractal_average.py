# -*- coding: utf-8 -*-
"""
BBA Fractal Operator 计算一组（或者一个BBA）的 seg_depth 阶分形
====================
实现对输入的基本概率分配（BBA）进行 **均等分形**：父焦元 A_j 的质量 m(A_j) 被平均分配到其所有非空子集 A_i 上，
每个子集获得的系数均为 1 / (2^{|A_j|} - 1)。当 |A_j| = 1 时，分母为 1，质量保持不变。
多阶分形依旧通过迭代一次分形实现，参数 h 仅表示迭代次数，本身不再出现在公式中。
定义 0 阶分形时直接返回原始 BBA，不支持负数分形。

在输出时保留四位小数。

使用方法
--------
在 data/examples 文件夹中放置了 CSV 文件，例如 Example_3_3.csv。

脚本可直接运行，无需命令行参数时默认使用 h=1（注意要转到目录 cd experiments）：
$ python fractal_average.py             # 等价于 h=1
或指定阶数 h：
$ python fractal_average.py 3          # h = 3

模块提供可导入接口，供其他脚本调用：
- powerset(s: FrozenSet[str]) -> Iterator[FrozenSet[str]]
- split_once(bba: Dict[FrozenSet[str], float], _h: int) -> Dict[FrozenSet[str], float]
- higher_order_bba(bba: Dict[FrozenSet[str], float], h: int) -> Dict[FrozenSet[str], float]
- bba_to_series(bba: Dict[FrozenSet[str], float], focal_order: List[str]) -> List[float]
- compute_fractal_df(bbas: List[Tuple[str, Dict[FrozenSet[str], float]]], focal_cols: List[str], h: int) -> pd.DataFrame

使用方法与命令行脚本一致。
"""

import itertools
import os
import sys
from typing import Dict, FrozenSet, List, Tuple, Iterator

import pandas as pd

from config import SEG_DEPTH
# 依赖本项目内现成工具函数 / 模块
from utility.io import load_bbas, format_set

__all__ = [
    'powerset', 'split_once', 'higher_order_bba', 'bba_to_series', 'compute_fractal_df'
]

# ------------------------------ 工具函数 ------------------------------ #

# 生成 s 的所有 *非空* 子集（包括自身）。
def powerset(s: FrozenSet[str]) -> Iterator[FrozenSet[str]]:
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
            # 空集质量不参与分形，直接保留
            new_bba[Aj] = new_bba.get(Aj, 0.0) + mass
            continue
        denom = (2 ** len(Aj)) - 1  # 非空子集数，此处为 2^Theta -1
        for Ai in powerset(Aj):
            factor = 1.0 / denom
            new_bba[Ai] = new_bba.get(Ai, 0.0) + mass * factor
    return new_bba


# 执行 h 阶均等分形（迭代 h 次 split_once）。
def higher_order_bba(bba: Dict[FrozenSet[str], float], h: int) -> Dict[FrozenSet[str], float]:
    if h == 0:
        return bba
    current = bba
    for _ in range(h):
        current = split_once(current, h)
    return current


# ------------------------------ I/O 与展示 ------------------------------ #

# 按照指定的焦元顺序，将 bba_dict 转为数值列表，用于构造 DataFrame 行
def bba_to_series(bba: Dict[FrozenSet[str], float], focal_order: List[str]) -> List[float]:
    data = {format_set(k): v for k, v in bba.items()}
    return [data.get(col, 0.0) for col in focal_order]


# 计算分形 BBA 的 DataFrame，包含名称和各焦元质量
def compute_fractal_df(
        bbas: List[Tuple[str, Dict[FrozenSet[str], float]]],
        focal_cols: List[str],
        h: int
) -> pd.DataFrame:
    rows = []
    for name, bba in bbas:
        fbba = higher_order_bba(bba, h)
        rows.append([f"{name}_h{h}"] + bba_to_series(fbba, focal_cols))
    return pd.DataFrame(rows, columns=["BBA"] + focal_cols).round(4)


# ------------------------------ 主函数 ------------------------------ #
if __name__ == '__main__':
    # 读取 h，允许 h=0，禁止负数
    try:
        # 默认分形深度可在 config.py 中调整
        h = int(sys.argv[1]) if len(sys.argv) > 1 else SEG_DEPTH
        if h < 0:
            raise ValueError
    except ValueError:
        print("参数 h 必须是不小于 0 的整数，示例：python fractal_average.py 2 或 python fractal_average.py")
        sys.exit(1)

    # 定位 CSV 数据
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(base_dir, '..', 'data', 'examples')

    # todo 分形的 BBA 数据位置，可按需修改
    csv_file = 'Example_3_3_3.csv'
    csv_path = os.path.normpath(os.path.join(csv_dir, csv_file))
    if not os.path.isfile(csv_path):
        print(f"找不到 CSV 文件: {csv_path}")
        sys.exit(1)

    # 载入 BBA
    df = pd.read_csv(csv_path)
    bbas, focal_cols = load_bbas(df)

    # 迭代 0~h 阶分形并打印结果
    out_k = None
    for k in range(0, h + 1):
        print(f"----- 分形结果 (h = {k}) -----")
        out_k = compute_fractal_df(bbas, focal_cols, k)
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
