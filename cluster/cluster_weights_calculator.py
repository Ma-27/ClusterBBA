# -*- coding: utf-8 -*-
"""Cluster Scale Weights Calculator
===============================

计算指定簇在所有焦元上的 \tilde n_p(A) 与 w_p(A)。

使用方式::

    python cluster_weights_calculator.py Clus1 m1 m2 m3 [--csv path/to/file.csv]

若无参数则使用脚本内置的示例簇和 CSV。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, FrozenSet, List, Tuple

import numpy as np
import pandas as pd

# 依赖本项目内现成工具函数 / 模块
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from cluster.one_cluster import initialize_cluster_from_csv
from config import SCALE_DELTA, SCALE_EPSILON
from utility.bba import BBA
from utility.io import format_set


# ------------------------------ 内部工具 ------------------------------ #

def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """数值稳定且支持向量化的 Sigmoid 函数"""
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def compute_votes_and_weights(
        clus: 'Cluster',
        *,
        delta: float = SCALE_DELTA,
        epsilon: float = SCALE_EPSILON,
) -> Tuple[
    Dict[FrozenSet[str], float],
    Dict[FrozenSet[str], float],
    Dict[FrozenSet[str], List[float]],
]:
    """返回 ``\tilde n_p(A)``, ``w_p(A)`` 及每个 BBA 的 ``h_\varepsilon``"""

    focal_sets = set()
    for _, bba in clus.get_bbas():
        focal_sets.update(bba.keys())

    # 严格禁止空集参与投票与权重计算
    focal_sets.discard(frozenset())

    ordered = sorted(focal_sets, key=BBA._set_sort_key)  # type: ignore[attr-defined]
    if not ordered or not clus.get_bbas():
        votes = {fs: 0.0 for fs in ordered}
        weights = {fs: 0.0 for fs in ordered}
        h_values = {fs: [] for fs in ordered}
        return votes, weights, h_values

    mass_mat = np.array([[b.get_mass(fs) for fs in ordered]
                         for _, b in clus.get_bbas()])
    # 归一化后送入到 Sigmoid 软化
    h_mat = _sigmoid((mass_mat - delta) / epsilon)
    votes_vec = h_mat.sum(axis=0)
    total = votes_vec.sum()
    weights_vec = votes_vec / total if total else np.zeros_like(votes_vec)

    votes = {fs: float(v) for fs, v in zip(ordered, votes_vec)}
    weights = {fs: float(w) for fs, w in zip(ordered, weights_vec)}
    h_values = {fs: h_mat[:, idx].tolist() for idx, fs in enumerate(ordered)}
    return votes, weights, h_values


# ------------------------------ 参数处理 ------------------------------ #

def _parse_cluster_spec(spec: str) -> Tuple[str, List[str]]:
    """解析 ``Name:a,b,c`` 形式的簇定义。"""
    if ':' not in spec:
        raise ValueError(f'簇定义 "{spec}" 缺少冒号分隔')
    name, bbas = spec.split(':', 1)
    items = [s.strip() for s in bbas.split(',') if s.strip()]
    if not items:
        raise ValueError(f'簇 {name} 未指定任何 BBA')
    return name.strip(), items


def _process_args(
        argv: List[str],
        default_csv: str,
        default_clusters: Dict[str, List[str]],
) -> Tuple[Dict[str, List[str]], str]:
    """解析命令行参数，兼容单簇与多簇格式"""

    if not argv:
        return default_clusters, default_csv

    csv_path = default_csv
    if '--csv' in argv:
        idx = argv.index('--csv')
        if idx + 1 >= len(argv):
            raise ValueError('参数错误: `--csv` 后缺少路径')
        csv_path = argv[idx + 1]
        del argv[idx:idx + 2]

    clusters: Dict[str, List[str]] = {}
    if '--clusters' in argv:
        idx = argv.index('--clusters')
        if idx + 1 >= len(argv):
            raise ValueError('参数错误: `--clusters` 后缺少定义字符串')
        spec = argv[idx + 1]
        del argv[idx:idx + 2]
        for part in spec.split(';'):
            part = part.strip()
            if part:
                name, items = _parse_cluster_spec(part)
                clusters[name] = items
    if argv:
        # 兼容旧格式：第一个参数簇名，之后为 BBA 名称列表
        name = argv.pop(0)
        bbas = argv
        if not bbas:
            raise ValueError('至少指定 1 个 BBA 名称。')
        clusters[name] = bbas

    if not clusters:
        clusters = default_clusters

    return clusters, csv_path


# ------------------------------ 主函数 ------------------------------ #

if __name__ == '__main__':  # pragma: no cover
    # todo 在这里更改数据集
    example_name = 'Example_3_2_3.csv'
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    default_csv = os.path.join(base_dir, 'data', 'examples', example_name)
    default_clusters = {'Clus3': ['m9', 'm10', 'm11', 'm12']}

    try:
        clusters, csv_path = _process_args(sys.argv[1:], default_csv, default_clusters)
        for name, bbas in clusters.items():
            clus = initialize_cluster_from_csv(name, bbas, csv_path)
            votes, weights, h_vals = compute_votes_and_weights(clus)

            ordered = sorted(votes.keys(), key=BBA._set_sort_key)  # type: ignore[attr-defined]
            fs_strings = [format_set(fs) for fs in ordered]

            df = pd.DataFrame({
                'tilde_n_p(A)': [votes[fs] for fs in ordered],
                'w_p(A)': [weights[fs] for fs in ordered],
            }, index=fs_strings).round(4)

            bba_names = [n for n, _ in clus.get_bbas()]
            h_data = {n: [h_vals[fs][idx] for fs in ordered]
                      for idx, n in enumerate(bba_names)}
            h_df = pd.DataFrame(h_data, index=fs_strings).round(4)

            print(f'----- Cluster "{name}" Scale Weights -----')
            print(df.to_string(float_format="%.4f"))
            print("h_epsilon values:")
            print(h_df.to_string(float_format="%.4f"))
            print()
    except Exception as e:
        print('ERROR:', e)
        sys.exit(1)
