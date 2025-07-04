# -*- coding: utf-8 -*-
"""
BBA Fractal Operator 计算一组（或者一个BBA）的 seg_depth 阶分形
====================
根据高阶 BBA 分形（Higher‑Order Basic Probability Assignment, HOBPA）公式 (Huang & Xiao, 2023)，
    h^{|A_j|-|A_i|} / \bigl((h+1)^{|A_j|}-h^{|A_j|}\bigr)
实现对输入的基本概率分配（BBA）进行 *h* 阶分形，并在输出时保留四位小数。

使用方法
--------
在 data/examples 文件夹中放置 CSV 文件，例如 Example_3_3.csv。

脚本可直接运行，无需命令行参数时默认使用 h=2（注意要转到目录 cd experiments）：
$ python fractal_hobpa.py             # 等价于 h=2
或指定阶数 h：
$ python fractal_hobpa.py 3          # h = 3

CSV 格式要求
------------
- 第一列标题必须为 **BBA**，表示每个质量函数的名称（如 m1, m2, ...）。
- 其余列标题用集合符号表示焦元，例如  `{A}`, `{B}`, `{A ∪ B}`。
- 每一行对应一个 BBA，单元格内为该焦元的质量值 (float)。

脚本将在控制台输出分形后的 BBA 表格。
多阶分形通过迭代一次分形得到。
定义 0 阶分形 (h=0) 时即返回原始 BBA，不进行任何分裂；不支持 h < 0。

脚本提供 Higher-Order Basic Probability Assignment (HOBPA) 的可导入接口，供其他脚本调用。

接口：
- powerset(s: FrozenSet[str]) -> Iterator[FrozenSet[str]]
- split_once(bba: BBA, h: int) -> BBA
- higher_order_bba(bba: BBA, h: int) -> BBA
 - compute_fractal_df(bbas: List[Tuple[str, BBA]], focal_cols: List[str], h: int) -> pd.DataFrame

示例：
```python
from fractal_hobpa import load_bbas, compute_fractal_df

import pandas as pd
df = pd.read_csv('data/examples/Example_2_3.csv')
bbas, focal_cols = load_bbas(df)
out_df = compute_fractal_df(bbas, focal_cols, h=3)
out_df.to_csv('result.csv', index=False)
```
"""

import itertools
import os
import sys
from typing import Dict, FrozenSet, List, Tuple, Iterator

import pandas as pd

from config import SEG_DEPTH
# 依赖本项目内现成工具函数 / 模块
from utility.bba import BBA
from utility.io import load_bbas

__all__ = [
    'powerset', 'split_once', 'higher_order_bba', 'compute_fractal_df'
]


# ------------------------------ 工具函数 ------------------------------ #

# 生成 s 的所有 *非空* 子集（包括自身）。
def powerset(s: FrozenSet[str]) -> Iterator[FrozenSet[str]]:
    items = list(s)
    for r in range(1, len(items) + 1):
        for combo in itertools.combinations(items, r):
            yield frozenset(combo)


# ------------------------------ 分形核心 ------------------------------ #

# 执行一次 1 阶分形，按论文公式分配质量
def split_once(bba: BBA, h: int) -> BBA:
    """执行一次 h 阶分形 (均等分配)"""
    new_bba: Dict[FrozenSet[str], float] = {}
    for Aj, mass in bba.items():
        # 单元素焦元或空集按原质量保留或均分到子集
        if len(Aj) == 0:
            # 空集质量不参与分形，直接保留
            new_bba[Aj] = new_bba.get(Aj, 0.0) + mass
            continue
        denom = (h + 1) ** len(Aj) - h ** len(Aj)  # 论文中公式分母
        for Ai in powerset(Aj):
            factor = h ** (len(Aj) - len(Ai)) / denom
            new_bba[Ai] = new_bba.get(Ai, 0.0) + mass * factor
    return BBA(new_bba)


# 执行 h 阶均等分形（迭代 h 次 split_once）。
def higher_order_bba(bba: BBA, h: int) -> BBA:
    # 如果 h=0，直接返回原始 BBA；h=1 时可直接调用一次分裂；更高阶通过迭代实现
    if h == 0:
        return bba
    if h == 1:
        return split_once(bba, 1)
    current: BBA = bba
    for _ in range(h):
        current = split_once(current, h)
    return current


# ------------------------------ I/O 与展示 ------------------------------ #

# 计算指定 h 阶分形结果并返回 DataFrame
def compute_fractal_df(
        bbas: List[Tuple[str, BBA]],
        focal_cols: List[str],
        h: int
) -> pd.DataFrame:
    """计算指定 h 阶分形结果并返回 DataFrame"""
    rows = []
    for name, bba in bbas:
        fbba = higher_order_bba(bba, h)
        rows.append([f"{name}_h{h}"] + fbba.to_series(focal_cols))
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
        print("参数 h 必须为非负整数，示例：python fractal_hobpa.py 2")
        sys.exit(1)

    # 定位 CSV 文件目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(base_dir, '..', 'data', 'examples')

    # todo 分形的 BBA 数据位置，可按需修改
    csv_file = 'Example_2_3.csv'
    csv_path = os.path.normpath(os.path.join(csv_dir, csv_file))
    if not os.path.isfile(csv_path):
        print(f"找不到 CSV 文件: {csv_path}")
        sys.exit(1)

    # 读取数据并生成 BBA 列表
    df = pd.read_csv(csv_path)
    bbas, focal_cols = load_bbas(df)

    # 遍历每一阶分形，打印结果，并在最后一阶保存
    out_df = None
    for k in range(0, h + 1):
        print(f"----- 分形结果 (h = {k}) -----")
        out_df = compute_fractal_df(bbas, focal_cols, k)
        # 打印第 k 轮结果
        print(out_df.to_string(index=False))

    # ------------------------------ 保存结果 ------------------------------ #
    # 创建输出结果文件夹 experiments_result，如果不存在则自动生成
    result_dir = os.path.normpath(os.path.join(base_dir, '..', 'experiments_result'))
    os.makedirs(result_dir, exist_ok=True)

    # 构造结果文件路径，文件名包含原始 CSV 名称与使用的 h 阶
    result_file = f"fractal_hobpa_{os.path.splitext(csv_file)[0]}_h{h}.csv"
    result_path = os.path.join(result_dir, result_file)
    # 将最后一阶结果保存为 CSV 文件，所有数值以四位小数格式输出
    out_df.to_csv(result_path, index=False, float_format='%.4f')
    print(f"\n结果已保存到: {result_path}")
