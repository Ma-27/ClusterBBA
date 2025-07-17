# -*- coding: utf-8 -*-
"""Example 3.7 随机顺序分簇评估
===============================
随机打乱 ``Example_3_7.csv`` 中 BBA 的加入顺序，
使用 :class:`MultiClusters` 在线贪心算法进行分簇，
并与文档给出的三簇真值对比，计算分类指标。
"""

from __future__ import annotations

import os
import random
import sys
from typing import Dict, List

from tqdm import tqdm

# 确保包导入路径指向项目根目录
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd
from sklearn.metrics import classification_report

# 依赖本项目内现成工具函数 / 模块
from cluster.multi_clusters import MultiClusters  # type: ignore
from utility.bba import BBA
from utility.io import load_bbas  # type: ignore
from config import SHUFFLE_TIMES, PROGRESS_NCOLS


def load_example_bbas(csv_path: str) -> Dict[str, BBA]:
    """读取 CSV 并返回名称到 BBA 的映射"""
    df = pd.read_csv(csv_path)
    bbas, _ = load_bbas(df)
    # 方便随机打乱，转换为字典形式
    return dict(bbas)


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


def evaluate_once(lookup: Dict[str, BBA], order: List[str], truth: List[set[str]]) -> tuple[bool, int]:
    """按照指定顺序插入 BBA，并判断簇划分是否与真值一致。"""

    mc = MultiClusters(debug=False)
    for n in order:
        mc.add_bba_by_reward(lookup[n])

    pred = [set(name for name, _ in clus.get_bbas()) for clus in mc._clusters.values()]
    correct = {frozenset(s) for s in pred} == {frozenset(s) for s in truth}
    return correct, len(pred)


if __name__ == "__main__":  # pragma: no cover
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "data", "examples", "Example_3_7.csv")
    lookup = load_example_bbas(csv_path)
    truth_clusters = get_ground_truth_clusters()
    names = list(lookup.keys())

    preds: List[int] = []  # 记录每次实验是否正确

    pbar = tqdm(range(SHUFFLE_TIMES), desc="实验进度", ncols=PROGRESS_NCOLS)
    for i in pbar:
        order = names.copy()
        # 每次随机打乱 BBA 的插入顺序
        random.shuffle(order)
        correct, count = evaluate_once(lookup, order, truth_clusters)
        preds.append(1 if correct else 0)
        pbar.set_postfix({"trial": i + 1, "clusters": count, "correct": correct})

    print(f"\n---- {SHUFFLE_TIMES}次实验综合评价 ----")
    y_true = [1] * SHUFFLE_TIMES
    # 输出三分类指标，1 表示全部分簇正确
    print(classification_report(y_true, preds, digits=3, zero_division=0))
