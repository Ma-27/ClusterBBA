# -*- coding: utf-8 -*-
"""Wine 数据集分簇评估脚本
=======================

统计按默认策略分簇后, 单元素簇比例是否过高, 用于粗略判断是否出现"过度分簇"。本脚本同样通过 :func:`construct_clusters_by_sequence` 构建簇结构。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cluster.multi_clusters import construct_clusters_by_sequence
from utility.io_application import load_application_dataset


def count_over_clustered(samples, threshold: float = 0.8) -> int:
    """返回单元素簇占比超过 ``threshold`` 的样本数量, 同时打印各样本的簇规模"""
    over_count = 0
    for idx, (_, bbas, _) in enumerate(samples, start=1):
        mc = construct_clusters_by_sequence(bbas, debug=False)
        sizes = [len(c.get_bbas()) for c in mc._clusters.values()]
        # 打印当前样本对应的分簇大小列表
        print(sizes)
        if not sizes:
            continue
        ratio = sum(1 for s in sizes if s == 1) / len(sizes)
        if ratio >= threshold:
            over_count += 1
    return over_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检查 Wine 数据集中的过度分簇比例")
    default_csv = Path(__file__).resolve().parents[1] / "data" / "kfold_xu_bba_wine.csv"
    parser.add_argument("--csv", type=str, default=str(default_csv), help="CSV 数据集路径")
    parser.add_argument("--threshold", type=float, default=0.8, help="判定过度分簇的单元素簇占比")
    parser.add_argument("--debug", action="store_true", help="仅评估前 2 条样本")
    args = parser.parse_args()

    samples = load_application_dataset(csv_path=args.csv, debug=args.debug)
    over = count_over_clustered(samples, threshold=args.threshold)
    print(f"Over-clustered samples: {over}/{len(samples)}")
    print(f"Over-clustered Ratio: {over / len(samples)}")
