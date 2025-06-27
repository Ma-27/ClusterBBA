# -*- coding: utf-8 -*-
"""
metric_test.py
=================
该模块提供对 BJS 度量(metric)性质的检验，并可作为包导入使用。

接口：
- test_nonnegativity(met_df) -> bool
- test_symmetry(met_df) -> bool
- test_triangle_inequality(met_df) -> bool
- run_all_tests(met_df) -> dict

示例：
```python
from divergence.bjs import load_bbas, metric_matrix
from divergence.bjs_metric_test import run_all_tests
import pandas as pd

df = pd.read_csv('data/examples/Example_3_3.csv')
bbas, _ = load_bbas(df)
met_df = metric_matrix(bbas)
results = run_all_tests(met_df)
print(results)
```
"""


# 1. 非负性
def test_nonnegativity(metric_df):
    # 验证度量性质：非负性，对称性，三角不等式
    labels = list(metric_df.index)
    print("\n----- 验证非负性 -----")

    neg_found = False
    for i in labels:
        for j in labels:
            if metric_df.loc[i, j] < 0:
                print(f"非负性失败: d({i},{j}) = {metric_df.loc[i, j]}")
                neg_found = True
    if not neg_found:
        print("所有距离均>=0，满足非负性")
    return not neg_found


# 2. 对称性
def test_symmetry(metric_df, tol=1e-8):
    labels = list(metric_df.index)
    print("\n----- 验证对称性 -----")
    asym_found = False
    for i in labels:
        for j in labels:
            if abs(metric_df.loc[i, j] - metric_df.loc[j, i]) > 1e-8:
                print(f"对称性失败: d({i},{j}) = {metric_df.loc[i, j]}, d({j},{i}) = {metric_df.loc[j, i]}")
                asym_found = True
    if not asym_found:
        print("所有距离满足对称性: d(i,j)=d(j,i)")
    return not asym_found


# 3. 三角不等式
def test_triangle_inequality(metric_df, tol=1e-8):
    labels = list(metric_df.index)
    print("\n----- 验证三角不等式 -----")
    tri_failed = False
    for i in labels:
        for j in labels:
            for k in labels:
                if metric_df.loc[i, k] > metric_df.loc[i, j] + metric_df.loc[j, k] + 1e-8:
                    print(
                        f"三角不等式失败: d({i},{k}) = {metric_df.loc[i, k]} > d({i},{j}) + d({j},{k}) = {metric_df.loc[i, j] + metric_df.loc[j, k]}"
                    )
                    tri_failed = True
    if not tri_failed:
        print("所有三角不等式均成立: d(i,k) <= d(i,j) + d(j,k)")
    return not tri_failed


def run_all_tests(met_df):
    """运行所有度量性质检验并返回结果字典"""
    return {
        'nonnegativity': test_nonnegativity(met_df),
        'symmetry': test_symmetry(met_df),
        'triangle_inequality': test_triangle_inequality(met_df)
    }
