# -*- coding: utf-8 -*-
"""多簇管理模块
================
提供 ``MultiClusters`` 类以统一管理多个 :class:`cluster.Cluster` ，并实现
基于 BBA 选簇流程。
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

from cluster.one_cluster import Cluster, initialize_empty_cluster  # type: ignore
from divergence.rd_ccjs import metric_matrix  # type: ignore
from mean.mean_divergence import average_divergence  # type: ignore

# 类型别名
BBA = Dict[frozenset[str], float]
NamedBBA = Tuple[str, BBA]

__all__ = [
    "MultiClusters",
]


class MultiClusters:
    """维护整个簇集合 `C` 的高层数据结构。"""

    def __init__(self) -> None:
        self._clusters: Dict[str, Cluster] = {}
        self._next_id: int = 1  # 自动生成新簇名称使用

    # ------------------------------------------------------------------
    # 基础操作
    # ------------------------------------------------------------------
    def add_cluster(self, cluster: Cluster) -> None:
        """注册一个新的簇。"""
        if cluster.name in self._clusters:
            raise ValueError(f"Duplicate cluster name: {cluster.name}")
        self._clusters[cluster.name] = cluster

    def get_cluster(self, name: str) -> Cluster:
        """获取指定名称的簇。"""
        if name not in self._clusters:
            raise KeyError(f"Cluster '{name}' not found")
        return self._clusters[name]

    def print_all_info(self) -> None:
        """打印所有簇的基本信息。"""
        print("========== MultiClusters Summary ==========")
        for clus in self._clusters.values():
            clus.print_info()

    def print_cluster_info(self, name: str) -> None:
        """打印单个簇的信息并返回该簇。"""
        clus = self.get_cluster(name)
        clus.print_info()

    # ------------------------------------------------------------------
    # 辅助工具
    # ------------------------------------------------------------------
    @staticmethod
    def _clone_cluster(src: Cluster) -> Cluster:
        """深拷贝一个簇（仅复制内部 BBA 数据）。"""
        clone = initialize_empty_cluster(name=src.name)
        for n, b in src.get_bbas():
            clone.add_bba(n, b, _init=True)
        return clone

    @staticmethod
    def _calc_intra_divergence(target: Cluster, clusters: List[Cluster]) -> Optional[float]:
        """按照定义计算 ``D_intra``。"""
        if len(target.get_bbas()) >= 2:
            return target.intra_divergence()
        if len(clusters) < 2:
            return None
        vals = [c.intra_divergence() for c in clusters if c != target and len(c.get_bbas()) >= 2]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    @staticmethod
    def _strategy_reward(clusters: List[Cluster], idx: int) -> Optional[float]:
        """计算策略 ``k`` 的收益 ``R_k``。``idx`` 为含新 BBA 的簇索引。"""
        K_k = len(clusters)
        # fixme 如果只有一个簇，则为了数值稳定性，规定平均散度为 1.0，硬性规定。
        if K_k == 1:
            avg_rd_cc = 1.0
        else:
            dist_df = metric_matrix(clusters)
            # debug
            print(f"\nthe distance between clusters by executing strategy {idx + 1}:")
            print(dist_df)
            avg_rd_cc = average_divergence(dist_df)
            print(f"the average distance between clusters by executing strategy {idx + 1} is : {avg_rd_cc:.4f}")

        d_intra = MultiClusters._calc_intra_divergence(clusters[idx], clusters)
        # fixme 特殊情况处理，为了数值稳定性
        if d_intra is None or d_intra == 0:
            return None
        reward = avg_rd_cc / (d_intra / K_k)
        return reward

    # ------------------------------------------------------------------
    # 选簇与入簇
    # ------------------------------------------------------------------
    def _evaluate_strategies(self, name: str, bba: BBA) -> Tuple[str, Optional[float]]:
        """内部使用，评估所有策略并返回最优簇名称。"""
        cluster_names = list(self._clusters.keys())
        best_reward: float = -1.0
        best_name = ""

        # 遍历现有簇，将 BBA 加入其中
        for idx, cname in enumerate(cluster_names):
            # 获取当前簇
            clones = [self._clone_cluster(c) for c in self._clusters.values()]
            clones[idx].add_bba(name, bba, _init=True)
            r = self._strategy_reward(clones, idx)
            print("Rewards of strategies:")
            if r is not None:
                print(f"Strategy {idx + 1} → {cname}: {r:.4f}")
                if r > best_reward:
                    best_reward = r
                    best_name = cname

        # 最后考虑新建簇策略
        new_clus = initialize_empty_cluster(name=f"Clus{self._next_id}")
        new_clus.add_bba(name, bba, _init=True)
        clones = [self._clone_cluster(c) for c in self._clusters.values()] + [new_clus]
        r_new = self._strategy_reward(clones, len(clones) - 1)
        if r_new is not None:
            print(f"Strategy {len(clones)} → New Cluster: {r_new:.4f}")
            if r_new > best_reward:
                best_reward = r_new
                best_name = new_clus.name
        return best_name, (best_reward if best_reward >= 0 else None)

    def add_bba_by_reward(self, name: str, bba: BBA) -> str:
        """根据收益自动选择簇并执行入簇，返回目标簇名。"""
        target_name, reward = self._evaluate_strategies(name, bba)
        # 无法评价时，新建簇，通常对应没有任何簇的情况
        if not target_name:
            target_name = f"Clus{self._next_id}"
            print("No one cluster found, no reward yet...")
            reward = None
        # 如果目标簇不存在，则说明初始化一个空簇并添加 BBA
        if target_name not in self._clusters:
            clus = initialize_empty_cluster(name=target_name)
            self._clusters[target_name] = clus
            self._next_id += 1
        clus = self._clusters[target_name]
        clus.add_bba(name, bba)
        print(
            f"\nSelected cluster: {target_name}, reward={reward:.4f}" if reward is not None else f"Selected cluster: {target_name}")
        print()
        return target_name
