# -*- coding: utf-8 -*-
"""Information Volume Calculator
===============================

基于最大 Deng 熵拆分 (MDESR) 计算 BBA 的信息体积。不断拆分直至相邻两次信息体积差小于给定阈值为止。
"""

import itertools
import os
import sys
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from config import IV_EPSILON
from entropy.deng_entropy import deng_entropy
from utility.bba import BBA
from utility.formula_labels import LABEL_I, LABEL_HI_M
from utility.io import load_bbas
from utility.plot_style import apply_style
from utility.plot_utils import highlight_overlapping_lines, savefig

apply_style()

__all__ = ["information_volume"]


# --------- 多重-BBA：一个焦元可出现多次（不合并） --------- #

@dataclass
class MassEntry:
    focal: FrozenSet[str]
    mass: float


def _deng_entropy(entries: List[MassEntry]) -> float:
    """对“多重-BBA”列表计算 Deng 熵（不合并同名焦元）"""
    return sum(
        deng_entropy(BBA({e.focal: e.mass}))
        for e in entries
        if e.mass != 0
    )


# ------------------------------ 信息体积计算 ------------------------------ #

def _split_once(entries: List[MassEntry]) -> List[MassEntry]:
    """按 MDESR 规则对当前列表中 |A|>1 的焦元做一次拆分"""
    new_entries: List[MassEntry] = []
    for e in entries:
        if len(e.focal) <= 1:
            # 单元素 or 空集：直接原样保留
            new_entries.append(e)
            continue

        # 当前待拆焦元的所有非空子集（含自身）
        elements = list(e.focal)
        subsets: List[FrozenSet[str]] = [
            frozenset(s)
            for r in range(1, len(elements) + 1)
            for s in itertools.combinations(elements, r)
        ]
        denom = sum(BBA.subset_cardinality(s) for s in subsets)

        # 依照局部 MDESR 权重拆分
        for sub in subsets:
            w = BBA.subset_cardinality(sub) / denom
            new_entries.append(MassEntry(sub, e.mass * w))

    return new_entries


# ------------------------------ 信息体积计算 ------------------------------ #

def information_volume(bba: BBA, epsilon: float = IV_EPSILON,
                       return_curve: bool = False) -> Tuple[float, List[float] | None]:
    """
    计算信息体积 HIV-mass(m)。
    若 `return_curve=True`，同时返回 Hi(m) 序列（用于画收敛曲线）。
    """
    # 把 utility.bba.BBA 转成多重-BBA 列表（每个焦元一条记录）
    entries = [MassEntry(f, m) for f, m in bba.items()]

    curve: List[float] = []
    prev = _deng_entropy(entries)
    curve.append(prev)

    while True:
        entries = _split_once(entries)
        curr = _deng_entropy(entries)
        curve.append(curr)

        if abs(curr - prev) < epsilon:
            return (curr, curve) if return_curve else (curr, None)

        prev = curr


# ------------------------------ 绘图函数 ------------------------------ #

def plot_curves(curves: Dict[str, List[float]], out_path: str) -> None:
    """绘制 Hi(m) 随迭代步数变化的曲线。"""
    fig, ax = plt.subplots()
    for name, vals in curves.items():
        steps = list(range(1, len(vals) + 1))
        ax.plot(steps, vals, label=name)
    ax.set_xlabel(LABEL_I)
    ax.set_ylabel(LABEL_HI_M)
    ax.set_title("Information Volume Convergence")
    # 强制让坐标轴从原点开始
    # ax.set_xlim(left=0)
    # ax.set_ylim(bottom=0)
    ax.legend()
    highlight_overlapping_lines(ax)
    savefig(fig, out_path)


# ------------------------------ 主函数 ------------------------------ #
if __name__ == "__main__":
    #  todo 数据集可以在此修改
    default_csv = "Example_information_volume.csv"
    csv_name = sys.argv[1] if len(sys.argv) > 1 else default_csv

    try:
        eps = float(sys.argv[2]) if len(sys.argv) > 2 else IV_EPSILON
        if eps <= 0:
            raise ValueError
    except ValueError:
        print("epsilon 必须为正数")
        sys.exit(1)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "data", "examples", csv_name)
    if not os.path.isfile(csv_path):
        print(f"找不到 CSV 文件: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    bbas, _ = load_bbas(df)

    results: List[Tuple[str, float]] = []
    curves: Dict[str, List[float]] = {}
    print("----- Information Volume 结果 -----")
    for bba in bbas:
        vol, curve = information_volume(bba, eps, return_curve=True)
        print(f"\n{bba.name}:")
        for i, h in enumerate(curve, start=1):
            print(f"  i={i:<2d} Hi(m)={h:.6f}")
        print(f"  >>> HIV-mass(m) = {vol:.6f}")
        results.append((bba.name, round(vol, 6)))
        curves[bba.name] = curve

    # 保存结果
    out_dir = os.path.join(base_dir, "..", "experiments_result")
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"information_volume_{os.path.splitext(csv_name)[0]}.csv"
    pd.DataFrame(results, columns=["BBA", "InformationVolume"]) \
        .to_csv(os.path.join(out_dir, out_file), index=False, float_format="%.6f")
    print(f"\n结果已保存到: {out_file}")

    fig_file = f"information_volume_{os.path.splitext(csv_name)[0]}_curve.png"
    plot_curves(curves, os.path.join(out_dir, fig_file))
    print(f"图像已保存到: {fig_file}")
