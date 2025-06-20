# -*- coding: utf-8 -*-
"""
Cluster‑Centroid Computation via Fractal Averaging
=================================================
本脚本演示**固定簇划分**情况下的簇分形平均 BBA（簇心）计算流程。

- **输入**：Example_3_3.csv（位于 ../data/examples）
- **簇划分**：预设 Clus1 = {m1,m2,m5}，Clus2 = {m3,m4}
- **输出**：在控制台打印每个簇以及对应的 \tilde m_F^(h)（四位小数），并将结果保存到 ../experiments_result/cluster_result_Example_3_3.csv

脚本保留 fractal_average.py 的 I/O 形式与注释风格，方便后续替换为**自动最优分簇算法**。
"""

import os
import sys
from typing import Dict, FrozenSet, List, Tuple

import pandas as pd

# ------------------------------ 依赖函数导入 ------------------------------ #
# 直接复用 fractal_average.py 中已实现的工具与分形核心
try:
    import fractal_average as fa
except ImportError as e:  # pragma: no cover
    print("未找到 fractal_average.py，请确保该文件与本脚本位于同一目录")
    sys.exit(1)

# ------------------------------ 固定簇配置 ------------------------------ #
# 预设簇及加入顺序；后续可替换为自动聚类算法
PRESET_CLUSTERS = {
    "Clus1": ["m1", "m2", "m5"],
    "Clus2": ["m3", "m4"],
}


# ------------------------------ 辅助函数 ------------------------------ #
# 将若干同阶分形 BBA 求算术平均，返回新的 BBA（dict[frozenset, float]）
def average_bbas(bbas: List[Dict[FrozenSet[str], float]]) -> Dict[FrozenSet[str], float]:
    if not bbas:
        return {}
    # 收集全集命题，确保所有焦元都在结果里
    all_keys = set().union(*bbas)
    avg_bba = {}
    for k in all_keys:
        avg_bba[k] = sum(b.get(k, 0.0) for b in bbas) / len(bbas)
    return avg_bba


# 格式化簇心输出，按指定焦元列顺序生成元组
def centroid_to_tuple(bba: Dict[FrozenSet[str], float], focal_cols: List[str]) -> Tuple[float, ...]:
    return tuple(round(bba.get(fa.parse_focal_set(col), 0.0), 4) for col in focal_cols)


# ------------------------------ 主流程 ------------------------------ #
if __name__ == "__main__":
    # 定位 CSV 数据
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.normpath(os.path.join(base_dir, "..", "data", "examples"))
    csv_file = "Example_3_3.csv"
    csv_path = os.path.join(csv_dir, csv_file)
    if not os.path.isfile(csv_path):
        print(f"找不到 CSV 文件: {csv_path}")
        sys.exit(1)

    # 载入 BBA 与焦元顺序
    df = pd.read_csv(csv_path)
    bbas, focal_cols = fa.load_bbas(df)

    # 构造名称到 BBA 的映射，便于索引
    bba_dict = {name: bba for name, bba in bbas}

    results = []  # 收集输出行

    # 遍历预设簇，计算对应簇心
    for idx, (clus_name, member_names) in enumerate(PRESET_CLUSTERS.items(), start=1):
        h = len(member_names) - 1  # 分形阶按 n_i-1 规则确定
        member_fractals = []
        for mem in member_names:
            original_bba = bba_dict[mem]
            fbba = fa.higher_order_bba(original_bba, h) if h > 0 else original_bba
            member_fractals.append(fbba)
        centroid = average_bbas(member_fractals)
        centroid_tuple = centroid_to_tuple(centroid, focal_cols)

        # 打印结果
        print(
            f"Clus{idx} = {{{', '.join(member_names)}}},  \u007E m_F{idx}^({h}) = {centroid_tuple}"
        )

        # 存储到结果列表
        results.append([
            clus_name,
            "{" + ", ".join(member_names) + "}",
            h,
            *centroid_tuple,
        ])

    # 保存到 experiments_result 目录
    result_dir = os.path.normpath(os.path.join(base_dir, "..", "experiments_result"))
    os.makedirs(result_dir, exist_ok=True)
    out_cols = [
        "Cluster",
        "Members",
        "h",
        *focal_cols,
    ]
    out_df = pd.DataFrame(results, columns=out_cols)
    out_path = os.path.join(result_dir, f"cluster_result_{os.path.splitext(csv_file)[0]}.csv")
    out_df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\n簇心结果已保存到: {out_path}")
