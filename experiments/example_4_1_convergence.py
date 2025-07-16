# -*- coding: utf-8 -*-
"""Example 4.1 迭代分形下簇心熵收敛实验
=====================================
从一条基线 BBA 出发，动态加入在该 BBA 上施加随机扰动的多条 BBA，观察簇心在最大熵分形下的 Deng 熵是否收敛，以及扰动幅度 ``delta`` 对收敛速度的影响。

可在命令行指定 CSV 文件名以替换默认 ``Example_4_1.csv``。
"""

from __future__ import annotations

import os
import sys
from collections import deque
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from cluster.one_cluster import initialize_empty_cluster
from entropy.deng_entropy import deng_entropy
from fractal.fractal_max_entropy import higher_order_bba
from utility.bba import BBA
from utility.formula_labels import LABEL_DENG_ENTROPY
from utility.io import load_bbas
from utility.plot_style import apply_style
from utility.plot_utils import highlight_overlapping_lines, savefig

apply_style()

# ------------------------------- 本实验用到的超参数 ------------------------------ #
# Example 4.1 默认扰动幅度列表
EXAMPLE_4_1_DELTAS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

# Example 4.1 默认扰动次数
PERTURBATION_BBA_NUMBERS: int = 100

# Example 4.1 默认Deng熵停止的阈值
PERTURBATION_STOP_EPSILON: float = 1e-3

# 连续 n 轮满足阈值才视为收敛，可随时调节
CONSECUTIVE_STOP_ROUNDS = 5


# ------------------------------ 基线设置 ------------------------------ #

def load_base_bba(csv_name: str) -> BBA:
    """从 ``data/examples`` 中读取名为 ``csv_name`` 的基线 BBA"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "data", "examples", csv_name)
    csv_path = os.path.normpath(csv_path)
    df = pd.read_csv(csv_path)
    bbas, _ = load_bbas(df)
    if not bbas:
        raise ValueError("CSV 文件为空，未找到 BBA")
    return bbas[0][1]


def has_converged(diff_history: deque[float],
                  eps: float,
                  need_rounds: int) -> bool:
    """
    最近 `need_rounds` 次 |ΔH| 是否都 < eps ？
    """
    return len(diff_history) == need_rounds and all(d < eps for d in diff_history)


# ------------------------------ 工具函数 ------------------------------ #

def perturb_bba(base: BBA, delta: float) -> BBA:
    """在 `base` 上加入 L¹-幅度为 `delta` 的随机扰动（一次性保证合法）。

    - 噪声零和，L¹=δ
    - 结果中所有焦元质量非负
    """
    # θ 的所有非空焦元
    focals = [fs for fs in base.theta_powerset() if fs]
    base_vec = np.array([base.get(fs, 0.0) for fs in focals])

    # 1) 选出可“donor”的焦元集合（质量 ≥ δ/2）
    donors = np.where(base_vec >= delta / 2)[0]
    if donors.size == 0:
        raise ValueError(f"δ/2 = {delta / 2} 太大，baseline 没有足够质量可被转移")

    donor_idx = np.random.choice(donors)
    receiver_idx = [i for i in range(len(focals)) if i != donor_idx]

    # 2) 在受益焦元上用 Dirichlet 随机分配正向质量 δ/2
    pos_alloc = np.random.dirichlet(np.ones(len(receiver_idx))) * (delta / 2)

    # 3) 构造新质量向量
    new_vec = base_vec.copy()
    new_vec[donor_idx] -= delta / 2
    new_vec[receiver_idx] += pos_alloc

    # 4) 封装为 BBA（去掉极小噪声）
    mass = {fs: float(v) for fs, v in zip(focals, new_vec) if v > 1e-12}
    return BBA(mass, frame=base.frame)


def limit_entropy(bba: BBA, eps: float = 1e-3, max_iter: int = 50) -> float:
    """计算 ``bba`` 在最大熵分形下的 Deng 熵极限"""
    prev = deng_entropy(bba)
    current = bba
    for _ in range(max_iter):
        current = higher_order_bba(current, 1)
        ent = deng_entropy(current)
        if abs(ent - prev) < eps:
            return ent
        prev = ent
    return ent


def _print_bba(name: str, bba: BBA) -> None:
    """将 BBA 质量表以 DataFrame 形式打印"""
    order = [BBA.format_set(fs) for fs in bba.theta_powerset()]
    df = pd.DataFrame([bba.to_series(order)], columns=order, index=[name])
    print(df.to_markdown(tablefmt="github", floatfmt=".4f"))


def run_single_delta(delta: float, rounds: int = PERTURBATION_BBA_NUMBERS, stop_eps: float = PERTURBATION_STOP_EPSILON,
                     need_rounds: int = CONSECUTIVE_STOP_ROUNDS) -> List[float]:
    """
    固定 δ 运行实验，返回每轮的极限熵；若 **连续 need_rounds 轮** 的 |Deng 熵增量| < stop_eps，则视为收敛，后续填 NaN 以便可视化时自动断线。
    """
    clus = initialize_empty_cluster("Clus")
    clus.add_bba("m0", BASE_BBA)

    prev_ent = limit_entropy(clus.get_centroid())
    entropies = [prev_ent]
    diff_hist: deque[float] = deque(maxlen=need_rounds)

    print("Round 0 (baseline):")
    _print_bba("m0", BASE_BBA)

    for r in range(1, rounds + 1):
        # 每轮生成一条扰动 BBA 并加入簇
        m = perturb_bba(BASE_BBA, delta)
        print(f"Round {r} delta={delta}")
        _print_bba(f"m{r}", m)
        clus.add_bba(f"m{r}", m)

        # 2) 计算当前簇心极限熵与增量
        ent = limit_entropy(clus.get_centroid())
        diff = abs(ent - prev_ent)
        diff_hist.append(diff)
        entropies.append(ent)
        prev_ent = ent

        # 3) 判断是否收敛，当增量持续低于阈值时提前退出
        if has_converged(diff_hist, stop_eps, need_rounds):
            entropies.extend([float("nan")] * (rounds - r))
            break

    # 若因收敛提前退出循环，仍保证长度一致
    if len(entropies) < rounds + 1:
        entropies.extend([float("nan")] * (rounds + 1 - len(entropies)))

    return entropies


# ------------------------------ 主函数 ------------------------------ #

if __name__ == "__main__":
    # todo 默认示例文件名，可根据实际情况修改
    default_name = "Example_4_1.csv"
    csv_name = sys.argv[1] if len(sys.argv) > 1 else default_name

    # 注意，这里只载入了第一个BBA。
    BASE_BBA = load_base_bba(csv_name)
    np.random.seed(0)

    records = {"round": list(range(0, PERTURBATION_BBA_NUMBERS + 1))}
    for d in EXAMPLE_4_1_DELTAS:
        records[f"delta={d}"] = run_single_delta(d, PERTURBATION_BBA_NUMBERS)

    df = pd.DataFrame(records)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.normpath(os.path.join(base_dir, "..", "experiments_result"))
    # 所有实验输出存放的位置
    os.makedirs(result_dir, exist_ok=True)

    dataset = os.path.splitext(os.path.basename(csv_name))[0]
    suffix = dataset.lower()
    if suffix.startswith("example_"):
        suffix = suffix[len("example_"):]

    # todo 在此可以修改路径信息。
    csv_path = os.path.join(result_dir, f"example_{suffix}_convergence_generation.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")

    fig, ax = plt.subplots()
    for d in EXAMPLE_4_1_DELTAS:
        ax.plot(df["round"], df[f"delta={d}"], label=f"$\\delta={d}$")
    ax.set_xlabel("Round")
    ax.set_ylabel(LABEL_DENG_ENTROPY)
    ax.set_title("Example 4.1 Fractal Convergence")
    ax.set_xlim(left=0)

    highlight_overlapping_lines(ax)
    ax.legend()

    fig_path = os.path.join(result_dir, f"example_{suffix}_convergence_curve.png")
    savefig(fig, fig_path)

    print(f"Results saved to: {csv_path}")
    print(f"Figure saved to: {fig_path}")
