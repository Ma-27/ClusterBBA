# -*- coding: utf-8 -*-
"""Example 3.7 随机顺序分簇评估
===============================
随机打乱 ``Example_3_7.csv`` 中 BBA 的加入顺序，使用 :func:`construct_clusters_by_sequence` 对 BBA 进行在线贪心分簇，并与三簇真值对比，计算分类指标。
"""

from __future__ import annotations

import os
import random
import sys
from typing import List, Dict

from tqdm import tqdm

# 确保包导入路径指向项目根目录
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd
from sklearn.metrics import classification_report

# 依赖本项目内现成工具函数 / 模块
from cluster.multi_clusters import construct_clusters_by_sequence, construct_clusters_by_sequence_dp  # type: ignore
from utility.bba import BBA
from utility.io import load_bbas  # type: ignore
from config import SHUFFLE_TIMES, PROGRESS_NCOLS


def load_example_bbas(csv_path: str) -> List[BBA]:
    """读取 CSV 并返回 ``[BBA, …]`` 列表"""
    df = pd.read_csv(csv_path)
    bbas, _ = load_bbas(df)
    return bbas


def get_ground_truth_labels() -> Dict[str, int]:
    """Example 3.7 中各 BBA 所属簇的真值标签"""
    return {
        "m1": 1, "m2": 1, "m3": 1, "m4": 1, "m5": 1,
        "m6": 2, "m7": 2, "m8": 2, "m9": 2,
        "m10": 2, "m11": 2, "m12": 2, "m13": 2,
        "m14": 3, "m15": 3, "m16": 3,
    }


def get_ground_truth_clusters() -> List[set[str]]:
    """根据真值标签返回簇成员集合列表。"""
    labels = get_ground_truth_labels()
    groups: Dict[int, set[str]] = {}
    for name, idx in labels.items():
        groups.setdefault(idx, set()).add(name)
    return list(groups.values())


def evaluate_once(order: List[BBA], truth: List[set[str]]) -> tuple[bool, List[int]]:
    """按照指定顺序插入 BBA, 返回是否正确及各簇规模"""

    mc = construct_clusters_by_sequence(order)

    clusters = list(mc._clusters.values())
    pred = [set(bba.name for bba in clus.get_bbas()) for clus in clusters]
    sizes = [len(clus.get_bbas()) for clus in clusters]
    correct = {frozenset(s) for s in pred} == {frozenset(s) for s in truth}
    return correct, sizes


def evaluate_accuracy(shuffle_times: int = SHUFFLE_TIMES) -> float:
    """重复打乱顺序评估分簇准确率"""

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "data", "examples", "Example_3_7.csv")
    bbas = load_example_bbas(csv_path)
    truth_clusters = get_ground_truth_clusters()

    correct_count = 0
    for _ in range(shuffle_times):
        order = bbas.copy()
        random.shuffle(order)
        correct, _ = evaluate_once(order, truth_clusters)
        if correct:
            correct_count += 1
    return correct_count / shuffle_times


def evaluate_accuracy(
        shuffle_times: int = SHUFFLE_TIMES,
        *,
        show_progress: bool = False,
        position: int = 0,
) -> float:
    """重复打乱顺序评估分簇准确率

    参数
    ----
    shuffle_times: 重复打乱评估的次数
    show_progress: 是否使用 tqdm 展示评估进度
    position: tqdm 进度条位置，便于与外部进度条配合
    """

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "data", "examples", "Example_3_7.csv")
    bbas = load_example_bbas(csv_path)
    truth_clusters = get_ground_truth_clusters()

    correct_count = 0
    iterable = range(shuffle_times)
    if show_progress:
        iterable = tqdm(
            iterable,
            desc="评估进度",
            ncols=PROGRESS_NCOLS,
            position=position,
            leave=False,
            dynamic_ncols=True,
        )
    for _ in iterable:
        order = bbas.copy()
        # 随机打乱 BBA 插入顺序
        random.shuffle(order)
        # 单次评估
        correct, _ = evaluate_once(order, truth_clusters)
        if correct:
            correct_count += 1
    return correct_count / shuffle_times


if __name__ == "__main__":  # pragma: no cover
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "data", "examples", "Example_3_7.csv")
    bbas = load_example_bbas(csv_path)
    truth_clusters = get_ground_truth_clusters()

    preds: List[int] = []  # 记录每次实验是否正确
    wrong_sizes: List[List[int]] = []  # 错误样本对应的簇规模

    pbar = tqdm(range(SHUFFLE_TIMES), desc="实验进度", ncols=PROGRESS_NCOLS)
    for i in pbar:
        order = bbas.copy()
        # 每次随机打乱 BBA 的插入顺序
        random.shuffle(order)
        correct, sizes = evaluate_once(order, truth_clusters)
        preds.append(1 if correct else 0)
        if not correct:
            wrong_sizes.append(sizes)
        pbar.set_postfix({"trial": i + 1, "clusters": len(sizes), "correct": correct})

    print(f"\n---- {SHUFFLE_TIMES}次实验综合评价 ----")
    y_true = [1] * SHUFFLE_TIMES
    # 输出三分类指标，1 表示全部分簇正确
    print(classification_report(y_true, preds, digits=3, zero_division=0))

    if wrong_sizes:
        print("\n---- 错误样本簇规模 ----")
        for s in wrong_sizes:
            print(s)
