import os
import sys
from typing import List

# 确保包导入路径指向项目根目录
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from cluster.multi_clusters import MultiClusters  # type: ignore
from divergence.rd_ccjs import divergence_matrix  # type: ignore
from mean.mean_divergence import average_divergence  # type: ignore
from utility.io import load_bbas  # type: ignore


# ------------------------------ 输入处理 ------------------------------ #
def _process_csv_path(argv: List[str], default_csv: str) -> str:
    return argv[0] if argv else default_csv


# -------------------------- 用于打印距离信息 -------------------------- #
def _print_distance_info(mc: MultiClusters) -> None:
    clusters = list(mc._clusters.values())
    if not clusters:
        return

    # 如果只有一个簇，需要通过二分估计，结果不稳定
    if len(clusters) == 1:
        # 只有一个簇时通过随机二分估计 RD_CCJS
        avg_rd = MultiClusters._avg_rd_cc_by_split(clusters[0])
        print("当前仅一个簇，距离矩阵省略")
    else:
        # 计算簇-簇距离矩阵并打印
        dist_df = divergence_matrix(clusters)
        print("簇间 RD_CCJS 距离矩阵：")
        print(dist_df.to_markdown(tablefmt="github", floatfmt=".4f"))
        avg_rd = average_divergence(dist_df)

    if avg_rd is None:
        print("平均 RD_CCJS: None")
    else:
        print(f"平均 RD_CCJS: {avg_rd:.4f}")

    print()

    # 打印每个簇的 D_intra
    for clus in clusters:
        di = clus.intra_divergence()
        if di is None:
            print(f"{clus.name} 的 D_intra: None")
        else:
            print(f"{clus.name} 的 D_intra: {di:.4f}")


# ------------------------------ BBA 动态入簇 ------------------------------ #
def bba_dynamic_adding(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    bbas, _ = load_bbas(df)
    mc = MultiClusters()
    for name, bba in bbas:
        print(f"------------------------------ Round: {name} ------------------------------")
        mc.add_bba_by_reward(name, bba)
        _print_distance_info(mc)
        print()


# ------------------------------ 主函数 ------------------------------ #
if __name__ == "__main__":  # pragma: no cover
    # todo 默认示例文件名，可根据实际情况修改
    example_name = "Example_1_3.csv"
    # 确定项目根目录：当前脚本位于 divergence/，故上溯一级
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    default_csv = os.path.join(base_dir, "data", "examples", example_name)
    if not os.path.isfile(default_csv):
        print(f"默认 CSV 文件不存在: {default_csv}")
        sys.exit(1)

    csv_path = _process_csv_path(sys.argv[1:], default_csv)

    try:
        bba_dynamic_adding(csv_path)
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
