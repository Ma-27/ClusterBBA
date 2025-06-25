# -*- coding: utf-8 -*-
"""
BJS Divergence Calculator Module
================================
提供 Belief Jensen–Shannon divergence (BJS) 的可导入接口，供其他脚本调用。

接口：
- parse_focal_set(cell) -> FrozenSet[str]
- load_bbas(df) -> (List[(name, bba_dict)], List[focal_col])
- bjs_divergence(m1, m2) -> float
- bjs_metric(m1, m2) -> float
- divergence_matrix(bbas) -> pd.DataFrame
- metric_matrix(bbas) -> pd.DataFrame
- save_csv(dist_df, out_path=None, default_name, label) -> None
- plot_heatmap(dist_df, out_path=None, default_name, title) -> None

示例：
```python
from bjs_distance import load_bbas, divergence_matrix, metric_matrix, save_csv, plot_heatmap
import pandas as pd

df = pd.read_csv('data/examples/Example_3_3.csv')
bbas, _ = load_bbas(df)
div_df = divergence_matrix(bbas)
met_df = metric_matrix(bbas)
save_csv(div_df, default_name='Example_3_3.csv', label='divergence')
save_csv(met_df, default_name='Example_3_3.csv', label='metric')
plot_heatmap(div_df, default_name='Example_3_3.csv', title='BJS Divergence Heatmap')
plot_heatmap(met_df, default_name='Example_3_3.csv', title='BJS Metric Heatmap')
```
"""

import math
import os
import sys
from typing import Dict, FrozenSet, List, Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------ 常量 ------------------------------ #
EPS = 1e-12  # 避免 log(0)


# ------------------------------ 接口函数 ------------------------------ #
# 解析焦元字符串为 frozenset，如 "{A ∪ B}" -> frozenset({'A','B'})
def parse_focal_set(cell: str) -> FrozenSet[str]:
    if cell.startswith("{") and cell.endswith("}"):
        cell = cell[1:-1]
    return frozenset(e.strip() for e in cell.split("∪") if e.strip())


# 从 DataFrame 构造 BBA 列表及焦元列顺序
def load_bbas(df: pd.DataFrame) -> Tuple[List[Tuple[str, Dict[FrozenSet[str], float]]], List[str]]:
    focal_cols = [c for c in df.columns if c != "BBA"]
    bbas: List[Tuple[str, Dict[FrozenSet[str], float]]] = []
    for _, row in df.iterrows():
        name = str(row.get("BBA", "m"))
        mass = {parse_focal_set(col): float(row[col]) for col in focal_cols}
        bbas.append((name, mass))
    return bbas, focal_cols


# 计算两条 BBA 之间的 BJS divergence
def bjs_divergence(
        m1: Dict[FrozenSet[str], float],
        m2: Dict[FrozenSet[str], float]
) -> float:
    keys = set(m1) | set(m2)
    div = 0.0
    for A in keys:
        p = m1.get(A, 0.0) or EPS
        q = m2.get(A, 0.0) or EPS
        m = 0.5 * (p + q)
        div += 0.5 * p * math.log(p / m, 2) + 0.5 * q * math.log(q / m, 2)
    # BJS 的取值范围应在 [0, 1]
    return max(0.0, min(div, 1.0))


# 对应度量 = 根号(divergence)
def bjs_metric(
        m1: Dict[FrozenSet[str], float],
        m2: Dict[FrozenSet[str], float]
) -> float:
    return math.sqrt(bjs_divergence(m1, m2))

# 生成对称 BJS 距离矩阵 DataFrame
def divergence_matrix(bbas: List[Tuple[str, Dict[FrozenSet[str], float]]]) -> pd.DataFrame:
    names = [n for n, _ in bbas]
    size = len(names)
    mat = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            d = bjs_divergence(bbas[i][1], bbas[j][1])
            mat[i][j] = mat[j][i] = d
    return pd.DataFrame(mat, index=names, columns=names).round(4)


# 生成 metric 矩阵
def metric_matrix(bbas: List[Tuple[str, Dict[FrozenSet[str], float]]]) -> pd.DataFrame:
    names = [n for n, _ in bbas]
    size = len(names)
    mat = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            m = bjs_metric(bbas[i][1], bbas[j][1])
            mat[i][j] = mat[j][i] = m
    return pd.DataFrame(mat, index=names, columns=names).round(4)


# 保存距离矩阵为 CSV 文件，可指定输出路径或使用默认文件名
def save_csv(
        dist_df: pd.DataFrame,
        out_path: Optional[str] = None,
        default_name: str = 'Example_3_3.csv',
        label: str = 'divergence',
        index_label: str = 'BBA'
) -> None:
    if out_path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        res = os.path.join(base, '..', 'experiments_result')
        os.makedirs(res, exist_ok=True)
        fname = f"bjs_{label}_{os.path.splitext(default_name)[0]}.csv"
        out_path = os.path.join(res, fname)
    dist_df.to_csv(out_path, float_format='%.4f', index_label=index_label)


# 绘制并保存热力图
def plot_heatmap(
        dist_df: pd.DataFrame,
        out_path: Optional[str] = None,
        default_name: str = 'Example_3_3.csv',
        title: Optional[str] = 'BJS Divergence Heatmap',
        label: str = 'divergence'
) -> None:
    if out_path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        res = os.path.join(base, '..', 'experiments_result')
        os.makedirs(res, exist_ok=True)
        fname = f"bjs_{label}_{os.path.splitext(default_name)[0]}.png"
        out_path = os.path.join(res, fname)
    fig, ax = plt.subplots()
    cax = ax.matshow(dist_df.values)
    fig.colorbar(cax)
    ax.set_xticks(range(len(dist_df)))
    ax.set_xticklabels(dist_df.columns, rotation=90)
    ax.set_yticks(range(len(dist_df)))
    ax.set_yticklabels(dist_df.index)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)


# ------------------------------ 主函数 ------------------------------ #
if __name__ == '__main__':
    # todo 默认示例文件，可以灵活修改
    default_name = 'Example_0_4.csv'
    # 处理命令行参数：CSV 文件名
    csv_name = sys.argv[1] if len(sys.argv) > 1 else default_name

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base, '..', 'data', 'examples', csv_name)
    if not os.path.isfile(csv_path):
        print(f"找不到 CSV 文件: {csv_path}")
        sys.exit(1)

    # 读取并处理 BBA
    df = pd.read_csv(csv_path)
    bbas, _ = load_bbas(df)

    # 计算距离矩阵
    div_df = divergence_matrix(bbas)
    met_df = metric_matrix(bbas)

    # ---------- 控制台输出 ---------- #
    print("\n----- BJS 距离矩阵 -----")
    print(div_df.to_string())
    print("\n----- BJS 度量矩阵 -----")
    print(met_df.to_string())

    # 保存并可视化
    save_csv(div_df, default_name=csv_name, label='divergence')
    print(f"Divergence结果CSV: experiments_result/bjs_{"divergence"}_{os.path.splitext(csv_name)[0]}.csv")
    # save_csv(met_df, default_name=csv_name, label='metric')
    # print(f"Metric结果CSV: experiments_result/bjs_{"metric"}_{os.path.splitext(csv_name)[0]}.csv")

    # plot_heatmap(div_df, default_name=csv_name, title='BJS Divergence Heatmap', label='divergence')
    # print(f"Divergence结果可视化: experiments_result/bjs_{"divergence"}_{os.path.splitext(csv_name)[0]}.png")
    # plot_heatmap(met_df, default_name=csv_name, title='BJS Metric Heatmap', label='metric')
    # print(f"Metric结果可视化: experiments_result/bjs_{"metric"}_{os.path.splitext(csv_name)[0]}.png")

    # 验证度量性质：非负性，对称性，三角不等式
    labels = list(met_df.index)

    # 1. 非负性
    print("\n----- 验证非负性 -----")
    neg_found = False
    for i in labels:
        for j in labels:
            if met_df.loc[i, j] < 0:
                print(f"非负性失败: d({i},{j}) = {met_df.loc[i, j]}")
                neg_found = True
    if not neg_found:
        print("所有距离均>=0，满足非负性")

    # 2. 对称性
    print("\n----- 验证对称性 -----")
    asym_found = False
    for i in labels:
        for j in labels:
            if abs(met_df.loc[i, j] - met_df.loc[j, i]) > 1e-8:
                print(f"对称性失败: d({i},{j}) = {met_df.loc[i, j]}, d({j},{i}) = {met_df.loc[j, i]}")
                asym_found = True
    if not asym_found:
        print("所有距离满足对称性: d(i,j)=d(j,i)")

    # 3. 三角不等式
    print("\n----- 验证三角不等式 -----")
    tri_failed = False
    for i in labels:
        for j in labels:
            for k in labels:
                if met_df.loc[i, k] > met_df.loc[i, j] + met_df.loc[j, k] + 1e-8:
                    print(
                        f"三角不等式失败: d({i},{k}) = {met_df.loc[i, k]} > d({i},{j}) + d({j},{k}) = {met_df.loc[i, j] + met_df.loc[j, k]}"
                    )
                    tri_failed = True
    if not tri_failed:
        print("所有三角不等式均成立: d(i,k) <= d(i,j) + d(j,k)")
