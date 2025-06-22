# -*- coding: utf-8 -*-
"""
BJS Divergence Calculator Module
================================
提供 Belief Jensen–Shannon divergence (BJS) 的可导入接口，供其他脚本调用。

- **接口**：
  - parse_focal_set(cell) -> FrozenSet[str]
  - load_bbas(df) -> (List[(name, bba_dict)], List[focal_col])
  - bjs_divergence(m1, m2) -> float
  - distance_matrix(bbas) -> pd.DataFrame

脚本用法（在 `experiments` 目录下执行）：
```bash
$ python bjs_distance.py                    # 默认读取 Example_3_3.csv
$ python bjs_distance.py Example_3_4.csv    # 指定文件
```
若文件不存在会给出提示。

示例：
```python
from bjs_distance import distance_matrix, load_bbas
import pandas as pd

df = pd.read_csv('data/examples/Example_3_3.csv')
bbas, cols = load_bbas(df)
mat = distance_matrix(bbas)
print(mat)
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


# 解析焦元字符串为 frozenset
# 如 "{A ∪ B}" -> frozenset({'A','B'})
def parse_focal_set(cell: str) -> FrozenSet[str]:
    if cell.startswith("{") and cell.endswith("}"):
        cell = cell[1:-1]
    items = [e.strip() for e in cell.split("∪") if e.strip()]
    return frozenset(items)


# 将 DataFrame 转换为 [(name, bba_dict), ...] 及焦元列顺序
def load_bbas(df: pd.DataFrame) -> Tuple[List[Tuple[str, Dict[FrozenSet[str], float]]], List[str]]:
    focal_cols = [c for c in df.columns if c != "BBA"]
    bbas: List[Tuple[str, Dict[FrozenSet[str], float]]] = []
    for _, row in df.iterrows():
        name = str(row.get("BBA", "m"))
        mass: Dict[FrozenSet[str], float] = {}
        for col in focal_cols:
            mass[parse_focal_set(col)] = float(row[col])
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
        p = m1.get(A, 0.0)
        q = m2.get(A, 0.0)
        # 避免 log(0)
        p = p if p > 0 else EPS
        q = q if q > 0 else EPS
        m = 0.5 * (p + q)
        div += 0.5 * p * math.log(p / m, 2) + 0.5 * q * math.log(q / m, 2)
    # BJS 的取值范围应在 [0, 1]
    return max(0.0, min(div, 1.0))


# 生成对称 BJS 距离矩阵 DataFrame
def distance_matrix(bbas: List[Tuple[str, Dict[FrozenSet[str], float]]]) -> pd.DataFrame:
    names = [name for name, _ in bbas]
    size = len(bbas)
    mat = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            d = bjs_divergence(bbas[i][1], bbas[j][1])
            mat[i][j] = mat[j][i] = d
    df = pd.DataFrame(mat, columns=names, index=names).round(4)
    return df

    # 可选：绘制并保存热力图


def plot_heatmap(
        dist_df: pd.DataFrame,
        out_path: str,
        title: Optional[str] = 'BJS Distance Matrix Heatmap'
) -> None:
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


# 脚本执行入口
if __name__ == '__main__':
    #  todo 可以在这里修改要求 BJS 的csv文件路径
    path = 'Example_3_3_3.csv'
    # path = 'Example_0.csv'

    # 处理命令行参数：CSV 文件名
    csv_name = sys.argv[1] if len(sys.argv) > 1 else path

    # 构造 CSV 路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.normpath(os.path.join(base_dir, '..', 'data', 'examples', csv_name))
    if not os.path.isfile(csv_path):
        print(f"找不到 CSV 文件: {csv_path}")
        sys.exit(1)

    # 读取 CSV 并解析 BBA
    df = pd.read_csv(csv_path)
    bbas, _ = load_bbas(df)

    # 计算距离矩阵
    dist_df = distance_matrix(bbas)

    # ---------- 控制台输出 ---------- #
    print("\n----- BJS 距离矩阵 -----")
    print(dist_df)

    # ---------- 保存 CSV ---------- #
    res_dir = os.path.join(base_dir, '..', 'experiments_result')
    os.makedirs(res_dir, exist_ok=True)
    csv_out = f"bjs_{os.path.splitext(csv_name)[0]}.csv"
    dist_df.to_csv(os.path.join(res_dir, csv_out), float_format='%.4f', index_label='BBA')
    print(f"结果 CSV: {res_dir}/{csv_out}")

    # ---------- 可视化热力图 ---------- #
    # 绘制距离矩阵热力图并保存
    img_out = f"bjs_{os.path.splitext(csv_name)[0]}.png"
    plot_heatmap(dist_df, os.path.join(res_dir, img_out))
    print(f"可视化图已保存到: {img_out}")
