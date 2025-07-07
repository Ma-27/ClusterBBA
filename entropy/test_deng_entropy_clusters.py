# -*- coding: utf-8 -*-
"""test_deng_entropy_clusters.py

仿照 ``test_multi_clusters.py``，动态加入 BBA 后计算每个簇心的 Deng 熵并可视化。
"""

import contextlib
import os
import sys
from io import StringIO
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from utility.formula_labels import LABEL_DENG_ENTROPY
from utility.plot_style import apply_style
from utility.plot_utils import savefig

apply_style()

# 确保包导入路径指向项目根目录
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from cluster.multi_clusters import MultiClusters  # type: ignore
from entropy.deng_entropy import deng_entropy
from utility.io import load_bbas  # type: ignore


def _process_csv_path(argv: List[str], default_csv: str) -> str:
    """处理命令行参数，返回 CSV 文件路径。"""
    return argv[0] if argv else default_csv


# 记录每个簇在每一步的熵值列表
EntropyHistory = Dict[str, List[float]]


def _record_entropies(step: int, mc: MultiClusters, history: EntropyHistory) -> None:
    """计算所有簇心的 Deng 熵并写入历史记录。"""
    current_clusters = list(mc._clusters.values())
    for clus in current_clusters:
        ent = deng_entropy(clus.get_centroid() or {})
        if clus.name not in history:
            history[clus.name] = [float('nan')] * (step - 1)
        history[clus.name].append(ent)
    # 对于不存在的簇填充 NaN
    for cname, vals in history.items():
        if len(vals) < step:
            vals.append(float('nan'))

    # 控制台打印
    print(f"Step {step} Centroid Deng Entropy:")
    for clus in current_clusters:
        ent = history[clus.name][-1]
        print(f"  {clus.name}: {ent:.4f}")
    print()


def _plot_history(history: EntropyHistory,
                  save_path: str | None = None,
                  show: bool = True) -> None:
    """绘制每个簇心 Deng 熵随时间变化的折线图。"""
    steps = range(1, max(len(v) for v in history.values()) + 1)
    for cname, vals in history.items():
        plt.plot(steps, vals, marker='o', label=cname)
    plt.xlabel('Step')
    plt.ylabel(LABEL_DENG_ENTROPY)
    plt.legend()
    if save_path:
        savefig(save_path)
    else:
        savefig('entropy_history.png')
    if show:
        plt.show()


# ------------------------------ 主函数 ------------------------------ #
if __name__ == '__main__':  # pragma: no cover
    # todo 默认示例，可按需修改
    example_name = 'Example_3_7.csv'

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    default_csv = os.path.join(base_dir, 'data', 'examples', example_name)
    if not os.path.isfile(default_csv):
        print(f'默认 CSV 文件不存在: {default_csv}')
        sys.exit(1)

    csv_path = _process_csv_path(sys.argv[1:], default_csv)

    # 读取 BBA 并依序加入 MultiClusters
    df = pd.read_csv(csv_path)
    bbas, _ = load_bbas(df)
    mc = MultiClusters()
    history: EntropyHistory = {}
    step = 0
    for name, bba in bbas:
        step += 1
        # 屏蔽 MultiClusters 内部的打印，仅关注熵值输出
        with contextlib.redirect_stdout(StringIO()):
            mc.add_bba_by_reward(name, bba)
        _record_entropies(step, mc, history)

    # 构建输出图像路径，文件名与其他实验保持一致风格
    result_dir = os.path.normpath(os.path.join(base_dir, 'experiments_result'))
    os.makedirs(result_dir, exist_ok=True)
    dataset = os.path.splitext(os.path.basename(csv_path))[0]
    suffix = dataset.lower()
    if suffix.startswith('example_'):
        suffix = suffix[len('example_'):]
    fig_path = os.path.join(result_dir, f"example_{suffix}_centroid_entropy.png")

    _plot_history(history, save_path=fig_path)
    print(f"Entropy history figure saved to: {fig_path}")
