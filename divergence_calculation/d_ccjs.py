# -*- coding: utf-8 -*-
"""
D_CCJS Divergence & Metric Calculator Module
============================================
提供 Cluster‑to‑Cluster Jensen–Shannon 散度 D_CCJS 及其对应度量 (Metric) 的可导入接口。

接口：
- parse_focal_set(cell) -> FrozenSet[str]
- load_bbas(df) -> (List[(name, bba_dict)], List[focal_col])
- d_ccjs_divergence(m_p, m_q, n_p, n_q) -> float
- d_ccjs_metric(m_p, m_q, n_p, n_q) -> float
- divergence_matrix(bbas, sizes) -> pd.DataFrame
- metric_matrix(bbas, sizes) -> pd.DataFrame
- save_csv(dist_df, out_path=None, default_name='Example_0_1.csv', label='divergence') -> None
- plot_heatmap(dist_df, out_path=None, default_name='Example_0_1.csv', title=None, label='divergence') -> None

脚本用法：
```bash
$ python d_ccjs.py [Example_0_1.csv]
```
示例：
```python
from d_ccjs import load_bbas, divergence_matrix, metric_matrix, save_csv, plot_heatmap
import pandas as pd

df = pd.read_csv('data/examples/Example_0_1.csv')
bbas, _ = load_bbas(df)
# 手动提供簇规模
sizes = {'Clus_1': 10, 'Clus_2': 7}
div_df = divergence_matrix(bbas, sizes)
met_df = metric_matrix(bbas, sizes)
print(div_df, met_df)
save_csv(div_df, default_name='Example_0_1.csv', label='divergence')
save_csv(met_df, default_name='Example_0_1.csv', label='metric')
plot_heatmap(div_df, default_name='Example_0_1.csv', title='D_CCJS Divergence Heatmap', label='divergence')
plot_heatmap(met_df, default_name='Example_0_1.csv', title='D_CCJS Metric Heatmap', label='metric')
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


# 计算两簇之间的 D_CCJS divergence（新定义）
def d_ccjs_divergence(
        m_p: Dict[FrozenSet[str], float],
        m_q: Dict[FrozenSet[str], float],
        n_p: int,
        n_q: int
) -> float:
    # 规模权重 w_p, w_q
    w_p = n_p / (n_p + n_q)
    w_q = n_q / (n_p + n_q)

    # 遍历所有焦元
    keys = set(m_p) | set(m_q)
    div = 0.0
    for A in keys:
        # fracional average BBA values
        p = m_p.get(A, 0.0) or EPS
        q = m_q.get(A, 0.0) or EPS
        # 混合分布 M(A) = 0.5 * p + 0.5 * q
        M = 0.5 * p + 0.5 * q
        # 按新公式累加
        div += w_p * p * math.log(p / M, 2) + w_q * q * math.log(q / M, 2)
    # 散度范围裁剪到 [0,1]
    return max(0.0, min(div, 1.0))


# fixme 对应度量 = 根号(divergence),考虑其他构造度量的方式
# 目前的构造度量 (Metric)：参考 RB 论文，将 CCJS 散度转换为度量
# RB_XY = sqrt(|D(XX) + D(YY) - 2 D(XY)| / 2)
def d_ccjs_metric(
        m_p: Dict[FrozenSet[str], float],
        m_q: Dict[FrozenSet[str], float],
        n_p: int,
        n_q: int
) -> float:
    # 第一种构造方式，直接取根号 CCJS
    # return math.sqrt(d_ccjs_divergence(m_p, m_q, n_p, n_q))

    # 第二种方式，参考 RB 论文构造度量
    d_pp = d_ccjs_divergence(m_p, m_p, n_p, n_p)
    d_qq = d_ccjs_divergence(m_q, m_q, n_q, n_q)
    d_pq = d_ccjs_divergence(m_p, m_q, n_p, n_q)
    return math.sqrt(abs(d_pp + d_qq - 2 * d_pq) / 2)


# 生成对称 D_CCJS divergence 矩阵
def divergence_matrix(
        bbas: List[Tuple[str, Dict[FrozenSet[str], float]]],
        sizes: Dict[str, int]
) -> pd.DataFrame:
    names = [name for name, _ in bbas]
    size = len(names)
    mat = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            mat[i][j] = mat[j][i] = d_ccjs_divergence(bbas[i][1], bbas[j][1], sizes[names[i]], sizes[names[j]])
    return pd.DataFrame(mat, index=names, columns=names).round(4)


# 生成 D_CCJS metric 矩阵
def metric_matrix(
        bbas: List[Tuple[str, Dict[FrozenSet[str], float]]],
        sizes: Dict[str, int]
) -> pd.DataFrame:
    names = [name for name, _ in bbas]
    size = len(names)
    mat = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            mat[i][j] = mat[j][i] = d_ccjs_metric(bbas[i][1], bbas[j][1], sizes[names[i]], sizes[names[j]])
    return pd.DataFrame(mat, index=names, columns=names).round(4)


# 保存矩阵为 CSV，可指定 divergence 或 metric
def save_csv(
        dist_df: pd.DataFrame,
        out_path: Optional[str] = None,
        default_name: str = 'Example_0_1.csv',
        label: str = 'divergence',
        index_label: str = 'Cluster'
) -> None:
    if out_path is None:
        # 保存到项目根目录下的 experiments_result
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        res = os.path.join(base, 'experiments_result')
        os.makedirs(res, exist_ok=True)
        fname = f"d_ccjs_{label}_{os.path.splitext(default_name)[0]}.csv"
        out_path = os.path.join(res, fname)
    dist_df.to_csv(out_path, float_format='%.4f', index_label=index_label)


# 绘制并保存热力图
def plot_heatmap(
        dist_df: pd.DataFrame,
        out_path: Optional[str] = None,
        default_name: str = 'Example_0_1.csv',
        title: Optional[str] = None,
        label: str = 'divergence'
) -> None:
    if title is None:
        title = f"D_CCJS {label.title()} Heatmap"
    if out_path is None:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        res = os.path.join(base, 'experiments_result')
        os.makedirs(res, exist_ok=True)
        fname = f"d_ccjs_{label}_{os.path.splitext(default_name)[0]}.png"
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
    # todo 默认示例文件，需要自定义
    default_name = 'Example_0_4.csv'
    csv_name = sys.argv[1] if len(sys.argv) > 1 else default_name

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_path = os.path.join(project_root, 'data', 'examples', csv_name)
    if not os.path.isfile(csv_path):
        print(f"找不到 CSV 文件: {csv_path}")
        sys.exit(1)

    # 读取并处理 BBA
    df = pd.read_csv(csv_path)
    bbas, _ = load_bbas(df)

    # todo 构造簇规模参数：请手动指定各簇的规模（根据实际 bbas 名称修改）
    # 例如，若有三个簇，名称分别为 'clus1','clus2','clus3'，规模为 1,2,3：
    sizes = {
        'm_F1_h3': 100,
        'm_F2_h3': 1,
        'm_F3_h3': 1,
        'm_F4_h3': 1,
    }

    # 计算并输出
    div_df = divergence_matrix(bbas, sizes)
    met_df = metric_matrix(bbas, sizes)

    print("\n----- D_CCJS Divergence 矩阵 -----")
    print(div_df.to_string())
    print("\n----- D_CCJS Metric 矩阵 -----")
    print(met_df.to_string())

    # 保存并可视化
    save_csv(div_df, default_name=csv_name, label='divergence')
    print(f"结果 CSV: experiments_result/d_ccjs_divergence_{os.path.splitext(csv_name)[0]}.csv")

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
