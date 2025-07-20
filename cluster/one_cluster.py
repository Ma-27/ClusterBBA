# -*- coding: utf-8 -*-
"""单簇操作模块
===============

提供 ``Cluster`` 类以管理单个簇并作为包导入的基础接口。

核心特性
--------
- 簇内 BBA 管理：内部维护一个无序、不重复的 BBA 列表，保留原始质量。
- 簇心 (Centroid) 更新：通过简单算术平均获得簇心，实时保持最新状态。
- BBA 入簇：支持动态添加新 BBA，自动触发簇心与簇内散度更新。
- 簇内散度：调用现有 `bjs.metric_matrix` 与 `mean_divergence.average_divergence` 计算簇内平均 BJS 距离。

命令行示例
^^^^^^^^^^
```bash
# 默认示例，无参数时根据代码中写好的默认情况运行
$ python one_cluster.py

# 指定 CSV 文件及 BBA 名称
$ python one_cluster.py ClusTest --csv path/to/Example.csv m1 m3 m4
```

导入接口示例
^^^^^^^^^^^^
```python
from cluster.one_cluster import Cluster, initialize_cluster_from_csv, initialize_empty_cluster

clus = initialize_empty_cluster(name='EmptyClus')
print('Empty cluster created:', clus)

clus = initialize_cluster_from_csv(
    name='ClusA',
    bba_names=['m1', 'm2', 'm5'],
    csv_path='data/examples/Example_3_3.csv'
)
print('Centroid:', clus.get_centroid())
print('Intra‑divergence:', clus.intra_divergence())
```
"""

import os
from typing import List, Optional

import pandas as pd

# 依赖本项目内现成工具函数 / 模块
from divergence.bjs import metric_matrix  # type: ignore
from fractal.fractal_max_entropy import higher_order_bba  # type: ignore
from mean.mean_bba import compute_avg_bba  # type: ignore
from mean.mean_divergence import average_divergence  # type: ignore
from utility.bba import BBA
from utility.io import load_bbas  # type: ignore

__all__ = [
    'Cluster',
    'initialize_cluster_from_csv',
    'initialize_empty_cluster',
]


# ------------------------------ 核心类 ------------------------------ #
# 该类表示单个质量函数簇。
class Cluster:
    def __init__(self, name: str, bbas: Optional[List[BBA]] = None):
        self.name: str = str(name)
        self._bbas: List[BBA] = []  # 原始 BBA 列表 (无序、不重复)
        self._centroid: Optional[BBA] = None  # 当前簇心 (延迟计算)
        self.h: int = 0  # 当前分形阶 h = n-1

        # 批量导入初始 BBA
        if bbas:
            for b in bbas:
                self.add_bba(b, _init=True)
        # 确保首次计算簇心
        self._sync_centroid()

    # 根据当前簇规模重新计算簇心，依据分形阶。
    def _sync_centroid(self) -> None:
        n = len(self._bbas)
        # 分形阶 h 与簇规模的关系为 h = n_i - 1
        self.h = max(n - 1, 0)
        if n == 0:
            self._centroid = {}
            return

        # 1) 对所有 BBA 做同阶分形
        fbba_list: List[BBA] = []
        for bba in self._bbas:
            fbba = higher_order_bba(bba, self.h)
            fbba_list.append(fbba)

        # 2) 求算术平均，根据公式推导
        self._centroid = compute_avg_bba(fbba_list)

    # ====================== 公开接口 ====================== #

    # 向簇中加入 新的 BBA 并同步簇心与平均散度。
    def add_bba(self, bba: BBA, _init: bool = False) -> Optional[float]:
        # 防止重复名称
        if any(existing.name == bba.name for existing in self._bbas):
            raise ValueError(f'Duplicate BBA "{bba.name}" in cluster "{self.name}"')
        # 递推更新簇心：新簇心由旧簇心与新 BBA 的同阶分形平均得到
        if self._centroid is None or len(self._bbas) == 0:
            # 第一个元素，退化为自身
            new_centroid = bba.copy()
            new_h = 0
        else:
            n_i = len(self._bbas)
            old_h = self.h
            # 1) 同步分形：F(old_centroid) 等价于 ``higher_order_bba(old_centroid, old_h + 1)``
            fractal_old = higher_order_bba(self._centroid, old_h + 1)
            # 2) 新元素的高阶分形 ``m_F^{(h)}``
            new_h = old_h + 1
            fractal_new = higher_order_bba(bba, new_h)
            # 3) 按 ``n_i/(n_i+1)`` 与 ``1/(n_i+1)`` 的权重合成新簇心
            new_centroid = {
                A: (n_i * fractal_old.get(A, 0.0) + fractal_new.get(A, 0.0)) / (n_i + 1)
                for A in set(fractal_old) | set(fractal_new)
            }
        # 将元素加入后更新内部状态
        self._bbas.append(bba)
        self.h = new_h
        self._centroid = new_centroid
        # 校验一致性
        self._sync_centroid()
        # if self._centroid != new_centroid:
        #     raise RuntimeError('Recursive update and sync results mismatch')
        if _init:
            return None
        return self.intra_divergence()

    # 返回当前簇心 (平均 BBA)。
    def get_centroid(self) -> Optional[BBA]:
        return self._centroid

    # 浅拷贝返回簇内全部 `(name, BBA)`。
    def get_bbas(self) -> List[BBA]:
        return list(self._bbas)

    # 将簇直接返回。
    def get(self):
        return self

    # 计算并返回簇内平均 BJS 距离。
    def intra_divergence(self) -> Optional[float]:
        # 若簇内不足 2 条 BBA，则返回 None（标志值），由 multi_clusters 脚本统一处理。
        if len(self._bbas) < 2:
            return None
        dist_df = metric_matrix(self._bbas)
        return average_divergence(dist_df)

    # 打印簇的基本信息：分形阶 h，元素个数 n_i，以及 BBA 质量 ASCII 表格。
    def print_info(self) -> None:
        print(f'----- Cluster: "{self.name}" Summary -----')
        print(f'Cluster "{self.name}":')
        print(f'Fractal order h = {self.h}')
        n_i = len(self._bbas)
        print(f'Number of BBA = {n_i}')
        if self.intra_divergence() is None:
            print('Empty or Single cluster, no intra-cluster divergence.')
        else:
            print(f'Intra‑cluster avg BJS distance: {self.intra_divergence():.4f}')
        # 打印簇中元素列表
        element_names = [b.name for b in self._bbas]
        formatted = ", ".join(f'"{n}"' for n in element_names)
        print(f'Elements: {formatted}')

        if isinstance(self._centroid, BBA):
            focal_sets = self._centroid.theta_powerset()
        else:
            focal_sets = []

        # 簇心质量表
        if self._centroid:
            print("Centroid mass table:")
            cent_data = {BBA.format_set(fs): [self._centroid.get(fs, 0.0)] for fs in focal_sets}
            cent_df = pd.DataFrame(cent_data, index=[self.name])
            print(cent_df.to_markdown(tablefmt="github", floatfmt=".4f"))

        # BBA 质量表
        print("BBA mass table:")
        data = {BBA.format_set(fs): [bba.get(fs, 0.0) for bba in self._bbas] for fs in focal_sets}
        df = pd.DataFrame(data, index=[b.name for b in self._bbas])
        # 打印 Markdown 表格
        print(df.to_markdown(tablefmt="github", floatfmt=".4f"))
        print()

    # 簇的字符串表示
    def __repr__(self) -> str:  # pragma: no cover
        return f'<Cluster {self.name!s} | {len(self._bbas)} BBA(s) | h={self.h}>'


# ------------------------------ 辅助函数 ------------------------------ #

# 根据 CSV 文件及 BBA 名称表构造一个簇。
def initialize_cluster_from_csv(name: str, bba_names: List[str], csv_path: str, *, strict: bool = True, ) -> Cluster:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'未找到 CSV 文件: {csv_path}')

    df = pd.read_csv(csv_path)
    # 利用 load_bbas 将 DataFrame 转换为 (name, bba_dict) 列表
    all_bbas, _ = load_bbas(df)

    # 按名称依次在列表中查找对应 BBA
    selected: List[BBA] = []
    missing: List[str] = []
    for n in bba_names:
        match = next((b for b in all_bbas if b.name == n), None)
        if match is None:
            missing.append(n)
        else:
            selected.append(match)

    # 严格模式下，若有缺失名称，直接抛出 KeyError
    if missing and strict:
        raise KeyError(f'BBA 名称未在 CSV 中找到: {missing}')

    # 构造并返回 Cluster 对象，初始化时批量添加 BBA
    return Cluster(name=name, bbas=selected)


# 初始化一个空簇，仅指定名称，不包含任何 BBA。
def initialize_empty_cluster(name: str) -> Cluster:
    return Cluster(name=name)
