# -*- coding: utf-8 -*-
"""test_d_intra.py

基于 ``cluster`` 的自动分簇流程，计算每个簇的 ``D_intra`` 随时间的变化，并提供是否处理边界情形的命令行开关。
"""

import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

# 依赖本项目内现成工具函数 / 模块
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from cluster.one_cluster import Cluster  # type: ignore
from cluster.multi_clusters import MultiClusters  # type: ignore
from utility.formula_labels import LABEL_D_INTRA
from utility.plot_style import apply_style
from utility.plot_utils import savefig
from utility.io import load_bbas  # type: ignore

apply_style()

DIHistory = Dict[str, List[float]]


def _calc_intra(clus: Cluster, clusters: List[Cluster], handle_boundary: bool) -> float:
    """使用 ``MultiClusters`` 的逻辑计算 ``D_intra``。

    ``handle_boundary`` 指示是否对单元素簇进行旧版边界处理。
    """
    val = MultiClusters._calc_intra_divergence(clus, clusters, handle_boundary)
    if val is None:
        return float('nan')
    return float(val)


def _record_history(step: int, clusters: Dict[str, Cluster], history: DIHistory, handle_boundary: bool) -> None:
    """记录当前各簇的 ``D_intra``。

    ``handle_boundary`` 控制是否在 ``D_intra`` 计算时处理边界。"""
    # ``clus_list`` 用于保持计算顺序一致
    clus_list = list(clusters.values())
    for clus in clus_list:
        # 计算指定簇在当前所有簇集合中的 ``D_intra``
        val = _calc_intra(clus, clus_list, handle_boundary)
        if clus.name not in history:
            # 若此前未出现该簇，需要补齐前面阶段的数据
            history[clus.name] = [float('nan')] * (step - 1)
        history[clus.name].append(val)
    for cname, vals in history.items():
        if len(vals) < step:
            vals.append(float('nan'))

    print(f"Step {step} D_intra:")
    for clus in clus_list:
        di_val = history[clus.name][-1]
        if di_val != di_val:
            print(f"  {clus.name}: None")
        else:
            print(f"  {clus.name}: {di_val:.4f}")
    print()


def _plot_history(history: DIHistory, save_path: str | None = None, show: bool = True) -> None:
    """绘制 ``D_intra`` 随时间变化的折线图。"""
    # 横轴为步骤序号，从 1 开始
    steps = range(1, max(len(v) for v in history.values()) + 1)
    for cname, vals in history.items():
        # 每个簇画一条折线
        plt.plot(steps, vals, marker='o', label=cname)
    plt.xlabel('Step')
    plt.ylabel(LABEL_D_INTRA)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()
    if save_path:
        savefig(save_path)
    else:
        savefig('d_intra_history.png')
    if show:
        plt.show()


if __name__ == "__main__":
    # todo 默认示例文件名，可按需要修改
    default_name = "Example_3_7_2.csv"
    # 命令行格式：python test_d_intra.py [CSV] [--handle-boundary]
    # 使用 "--handle-boundary" 或 "-b" 开启边界处理逻辑
    args = sys.argv[1:]

    # todo 在这里测验，是否使用 D_intra 的替代策略。
    handle_boundary = True

    csv_name = default_name
    for arg in args:
        if arg in ("-b", "--handle-boundary"):
            handle_boundary = True
        else:
            csv_name = arg

    # 确定项目根目录：当前脚本位于 divergence/，故上溯一级
    base = os.path.dirname(os.path.abspath(__file__))
    # 构造数据文件路径（相对于项目根）
    csv_path = os.path.join(base, "..", "data", "examples", csv_name)
    if not os.path.isfile(csv_path):
        print(f"找不到 CSV 文件: {csv_path}")
        sys.exit(1)

    # 读取 BBA
    df = pd.read_csv(csv_path)
    all_bbas, _ = load_bbas(df)

    # 动态分簇：使用 ``MultiClusters`` 根据收益自动选择簇。
    mc = MultiClusters(debug=False)

    # 用于记录各步骤的 ``D_intra`` 结果
    history: DIHistory = {}
    step = 0

    # 按 CSV 顺序动态添加 BBA，并记录每一步的 ``D_intra`` 变化
    for bba in all_bbas:
        step += 1
        # 根据当前簇划分策略选择最佳簇加入
        mc.add_bba_by_reward(bba)
        # 将当前结果写入历史
        _record_history(step, mc._clusters, history, handle_boundary)

    # 获取最终簇划分结果
    clusters = mc._clusters
    clus_list = list(clusters.values())

    print("\n----- D_intra per Cluster -----")
    results = []  # 保存最终 ``D_intra`` 数值
    for clus in clus_list:
        di = MultiClusters._calc_intra_divergence(clus, clus_list, handle_boundary)
        if di is None:
            print(f"{clus.name}: None")
            results.append([None])
        else:
            print(f"{clus.name}: {di:.4f}")
            results.append([di])

    df_res = pd.DataFrame(results, index=[c.name for c in clus_list], columns=["D_intra"])

    # 保存结果 CSV
    out_dir = os.path.abspath(os.path.join(base, "..", "experiments_result"))
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"d_intra_{os.path.splitext(csv_name)[0]}.csv")
    df_res.to_csv(out_csv, float_format="%.4f", index_label="Cluster")
    print(f"结果 CSV: experiments_result/{os.path.basename(out_csv)}")

    dataset = os.path.splitext(os.path.basename(csv_name))[0]
    suffix = dataset.lower()
    if suffix.startswith('example_'):
        suffix = suffix[len('example_'):]
    # 绘制 ``D_intra`` 变化并保存
    fig_path = os.path.join(out_dir, f'example_{suffix}_d_intra_history.png')
    _plot_history(history, save_path=fig_path)
    print(f'History figure saved to: {fig_path}')
