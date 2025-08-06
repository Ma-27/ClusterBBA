# -*- coding: utf-8 -*-
"""test_bba.py
===============

展示 :mod:`utility.bba` 中主要接口的用法。现在支持从 ``data/examples``
文件夹读取 CSV，对其中每条 BBA 逐一演示常用操作。
"""

import os
import pprint
import sys

# 确保包导入路径指向项目根目录
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from utility.bba import BBA
from utility.probability import pignistic, argmax
from utility.io import load_bbas


def test_bba(bba: BBA) -> None:
    """测试单条 BBA 的基本操作"""
    print(f"--------------- {bba.name} ---------------")

    print("--- BBA 初始化 & 基本属性 ---")
    print("识别框架:", sorted(bba.frame))
    print("质量和:", bba.total_mass())
    print("通过验证:" if bba.validate() else "未通过")

    print("\n焦元列表:", [BBA.format_set(fs) for fs in bba.focal_sets()])
    print("Theta 非空子集数量:", bba.theta_cardinality)

    # 集合运算示例：尝试取前两个子集
    subsets = bba.theta_powerset(include_empty=False)
    print("\n--- 集合运算 ---")
    if len(subsets) >= 2:
        A, B = subsets[0], subsets[1]
        union = BBA.union(A, B)
        intersection = BBA.intersection(A, B)
        print("A:", BBA.format_set(A), "B:", BBA.format_set(B))
        print("A ∪ B:", BBA.format_set(union))
        print("A ∩ B:", BBA.format_set(intersection))
        print("A ⊆ union?:", BBA.is_subset(A, union))
        print("union ⊇ A?:", BBA.is_superset(union, A))
        print("并集的元素个数:", BBA.cardinality(union))
        print("所有包含 A 的集合:", [BBA.format_set(s) for s in bba.supersets(A)])
        print("A 的子集:", [BBA.format_set(s) for s in bba.subsets_of(A, include_empty=True)])
    else:
        print("识别框架子集不足两项，跳过集合运算演示")

    print("\n--- Powerset ---")
    print([BBA.format_set(fs) for fs in bba.theta_powerset(include_empty=True)])

    print("\n--- 导入导出 ---")
    formatted = bba.to_formatted_dict()
    pprint.pprint(formatted)
    parsed = {BBA.parse_focal_set(k): v for k, v in formatted.items()}
    bba2 = BBA(parsed)
    print("解析后与原始相同:", bba2 == bba)
    print()

    print("--- Pignistic 概率 ---")
    prob = pignistic(bba)
    order = [BBA.format_set(fs) for fs in sorted(prob.keys(), key=BBA._set_sort_key)]
    row = [bba.name] + prob.to_series(order)
    df_prob = pd.DataFrame([row], columns=["BBA"] + order).round(4)
    print(df_prob.to_markdown(tablefmt="github", floatfmt=".4f"))

    pred_fs, pred_p = argmax(prob)
    print()
    print("预测命题:", BBA.format_set(pred_fs), f"概率:{pred_p:.4f}")
    print()


if __name__ == "__main__":  # pragma: no cover
    """从 CSV 加载 BBA 并逐一演示"""
    # todo 默认示例文件，可以灵活修改
    default_name = "Example_3_3.csv"
    csv_name = sys.argv[1] if len(sys.argv) > 1 else default_name

    # 确定项目根目录：当前脚本位于 utility/，故上溯一级
    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, "..", "data", "examples", csv_name)
    if not os.path.isfile(csv_path):
        print(f"找不到 CSV 文件: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    bbas, _ = load_bbas(df)

    for bba in bbas:
        test_bba(bba)
