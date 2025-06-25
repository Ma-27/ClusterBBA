# -*- coding: utf-8 -*-
"""
CCJS Divergence Calculator Module
================================
提供 Cluster‑to‑Cluster Jensen–Shannon divergence (CCJS) 的可导入接口，供其他脚本调用。

接口：
- parse_focal_set(cell) -> FrozenSet[str]
- load_bbas(df) -> (List[(name, bba_dict)], List[focal_col])
- ccjs_divergence(m_p, m_q, n_p, n_q) -> float
- distance_matrix(bbas, sizes) -> pd.DataFrame
- save_csv(dist_df, out_path=None, default_name='Example_0_1.csv') -> None
- plot_heatmap(dist_df, out_path=None, default_name='Example_0_1.csv', title) -> None

脚本用法：
```bash
$ python ccjs.py [Example_0_1.csv]
```
若文件不存在会给出提示。

示例：
```python
from ccjs import load_bbas, distance_matrix, save_csv, plot_heatmap
import pandas as pd

df = pd.read_csv('data/examples/Example_0_1.csv')
bbas, _ = load_bbas(df)
# 手动提供簇规模
sizes = {'Clus_1': 10, 'Clus_2': 7}
df_dist = distance_matrix(bbas, sizes)
print(df_dist)
save_csv(df_dist)                        # 默认使用 Example_0_1.csv 构建路径
plot_heatmap(df_dist)                    # 默认保存 heatmap
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


# 计算两簇之间的 CCJS divergence
def ccjs_divergence(
        m_p: Dict[FrozenSet[str], float],
        m_q: Dict[FrozenSet[str], float],
        n_p: int,
        n_q: int
) -> float:
    # 规模权重 w_p, w_q
    w_p = n_p / (n_p + n_q)
    w_q = n_q / (n_p + n_q)

    keys = set(m_p) | set(m_q)
    div = 0.0
    for A in keys:
        p = m_p.get(A, 0.0) or EPS
        q = m_q.get(A, 0.0) or EPS
        m_mix = w_p * p + w_q * q
        div += 0.5 * p * math.log(p / m_mix, 2) + 0.5 * q * math.log(q / m_mix, 2)
    # CCJS 的取值范围应在 [0, 1]
    return max(0.0, min(div, 1.0))


# 生成对称 CCJS 距离矩阵 DataFrame
def distance_matrix(
        bbas: List[Tuple[str, Dict[FrozenSet[str], float]]],
        sizes: Dict[str, int]
) -> pd.DataFrame:
    names = [name for name, _ in bbas]
    size = len(names)
    mat = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1, size):
            m_p, m_q = bbas[i][1], bbas[j][1]
            n_p, n_q = sizes[names[i]], sizes[names[j]]
            mat[i][j] = mat[j][i] = ccjs_divergence(m_p, m_q, n_p, n_q)
    return pd.DataFrame(mat, index=names, columns=names).round(4)


# 保存距离矩阵为 CSV 文件，可指定输出路径或使用默认文件名
def save_csv(
        dist_df: pd.DataFrame,
        out_path: Optional[str] = None,
        default_name: str = 'Example_0_1.csv',
        index_label: str = 'Cluster'
) -> None:
    if out_path is None:
        # 保存到项目根目录下的 experiments_result
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        res = os.path.join(base, 'experiments_result')
        os.makedirs(res, exist_ok=True)
        fname = f"ccjs_{os.path.splitext(default_name)[0]}.csv"
        out_path = os.path.join(res, fname)
    dist_df.to_csv(out_path, float_format='%.4f', index_label=index_label)


# 绘制并保存热力图
def plot_heatmap(
        dist_df: pd.DataFrame,
        out_path: Optional[str] = None,
        default_name: str = 'Example_0_1.csv',
        title: Optional[str] = 'CCJS Distance Matrix Heatmap'
) -> None:
    if out_path is None:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        res = os.path.join(base, 'experiments_result')
        os.makedirs(res, exist_ok=True)
        fname = f"ccjs_{os.path.splitext(default_name)[0]}.png"
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
    default_name = 'Example_0_1.csv'
    csv_name = sys.argv[1] if len(sys.argv) > 1 else default_name

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_path = os.path.join(project_root, 'data', 'examples', csv_name)
    if not os.path.isfile(csv_path):
        print(f"找不到 CSV 文件: {csv_path}")
        sys.exit(1)

    # 读取并处理 BBA
    df = pd.read_csv(csv_path)
    bbas, _ = load_bbas(df)

    # ---------------- 交互式获取簇规模 n ---------------- #
    sizes: Dict[str, int] = {}
    print("请输入每个簇的规模 n (正整数)：")
    for name, _ in bbas:
        while True:
            try:
                n = int(input(f"  - 簇 {name} 的 n = "))
                if n <= 0:
                    raise ValueError
                sizes[name] = n
                break
            except ValueError:
                print("    输入无效，请输入正整数！")

    # 计算距离矩阵并输出
    dist_df = distance_matrix(bbas, sizes)

    # ---------- 控制台输出 ---------- #
    print("\n----- CCJS 距离矩阵 -----")
    print(dist_df.to_string())

    # 保存并可视化
    save_csv(dist_df, default_name=csv_name)
    print(f"结果 CSV: experiments_result/ccjs_{os.path.splitext(csv_name)[0]}.csv")

    plot_heatmap(dist_df, default_name=csv_name)
    print(f"可视化图已保存到: experiments_result/ccjs_{os.path.splitext(csv_name)[0]}.png")
