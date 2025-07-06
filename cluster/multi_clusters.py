# -*- coding: utf-8 -*-
"""多簇管理模块
================
提供 ``MultiClusters`` 类以统一管理多个 :class:`cluster.Cluster` ，并实现
基于 BBA 选簇流程。
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple, Optional

# 依赖本项目内现成工具函数 / 模块
from cluster.one_cluster import Cluster, initialize_empty_cluster  # type: ignore
from config import THRESHOLD_BJS, SPLIT_TIMES, INTRA_EPS
from divergence.bjs import bjs_metric
from divergence.rd_ccjs import divergence_matrix  # type: ignore
from mean.mean_divergence import average_divergence  # type: ignore
from utility.bba import BBA

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
        # 如果目标簇本身就有两个或更多 BBA，则直接计算其内部散度
        if len(target.get_bbas()) >= 2:
            return target.intra_divergence()
        # 如果目标簇只有一个 BBA，并且还只有一个簇，则这种情况不需要此函数考虑，而是需要启动条件考虑。
        if len(clusters) < 2:
            return None
        # 在目标簇只有一条 BBA 时，会使用其他簇的平均簇内散度作为 d_intra（如果其他簇中，至少有一个簇具备2个及以上BBAs）。
        vals = [c.intra_divergence() for c in clusters if c != target and len(c.get_bbas()) >= 2]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    @staticmethod
    def _avg_rd_cc_by_split(cluster: Cluster, times: int = SPLIT_TIMES) -> Optional[float]:
        """将单簇随机二分多次，估计 ``RD_CCJS`` 均值。"""
        bbas = cluster.get_bbas()
        # 如果并非一个多元簇，则无法进行二分估计。这是为了防止单簇单元 (K=1, n1=1) 情形下的错误。
        if len(bbas) < 2:
            return None
        sum_rd_cc: List[float] = []
        for _ in range(times):
            idxs = list(range(len(bbas)))
            random.shuffle(idxs)
            cut = max(len(bbas) // 2, 1)
            left = [bbas[i] for i in idxs[:cut]]
            right = [bbas[i] for i in idxs[cut:]]
            # 保证两侧都有元素
            if not left or not right:
                continue
            clus_l = initialize_empty_cluster(name="L")
            clus_r = initialize_empty_cluster(name="R")
            for n, b in left:
                clus_l.add_bba(n, b, _init=True)
            for n, b in right:
                clus_r.add_bba(n, b, _init=True)
            dist_df = divergence_matrix([clus_l, clus_r])
            sum_rd_cc.append(average_divergence(dist_df))
        #
        if not sum_rd_cc:
            return None
        return sum(sum_rd_cc) / len(sum_rd_cc)

    @staticmethod
    def _strategy_reward(clusters: List[Cluster], idx: int, all_singletons: bool = False) -> Optional[float]:
        """计算策略 ``k`` 的收益 ``R_k``。``idx`` 为含新 BBA 的簇索引。
        ``all_singletons`` 标记在调用时是否处于多簇全为单元簇的边界情况，如果是的话，则要执行非常特殊的处理。"""
        K_k = len(clusters)
        # 单簇多元情形：通过随机二分估计 ``RD_CCJS``
        if K_k == 1:
            avg_rd_cc = MultiClusters._avg_rd_cc_by_split(clusters[0]) or 1.0
            # print(f"\nThe RD_CCJS distance between clusters by executing strategy {idx + 1} is {avg_rd_cc:.4f}")
        else:
            dist_df = divergence_matrix(clusters)
            # debug
            # print(f"\nThe RD_CCJS distance between clusters by executing strategy {idx + 1}:")
            # print(dist_df)
            avg_rd_cc = average_divergence(dist_df)
            # print(f"the average distance between clusters by executing strategy {idx + 1} is : {avg_rd_cc:.4f}")

        # 计算所有簇的平均 D_intra
        if all_singletons:
            # print("All clusters are singletons, using INTRA_EPS for D_intra.")
            d_intras = [INTRA_EPS for _ in clusters]
        else:
            # 正常情况，计算每个簇的 ``D_intra`` 并求平均
            d_intras: List[float] = []
            for clus in clusters:
                di = MultiClusters._calc_intra_divergence(clus, clusters)
                if di is None:
                    return None
                d_intras.append(di)
        # 分母，平均 D_intra
        avg_d_intra = sum(d_intras) / K_k

        # 防止除以0，为了数值稳定性
        if avg_d_intra == 0:
            return None

        reward = avg_rd_cc / avg_d_intra
        return reward

    # ------------------------------------------------------------------
    # 选簇与入簇
    # ------------------------------------------------------------------
    def _evaluate_strategies(self, name: str, bba: BBA) -> Tuple[str, Optional[float]]:
        """内部使用，评估所有策略并返回最优簇名称。"""
        cluster_names = list(self._clusters.keys())
        best_reward: float = -1.0
        best_name = ""
        print("Rewards of strategies:")

        # 判断是否处于多簇全为单元簇的边界情形
        all_singletons = len(self._clusters) >= 2 and all(
            len(c.get_bbas()) == 1 for c in self._clusters.values()
        )

        # 遍历现有簇，将 BBA 加入其中
        for idx, cname in enumerate(cluster_names):
            # 获取当前簇
            clones = [self._clone_cluster(c) for c in self._clusters.values()]
            clones[idx].add_bba(name, bba, _init=True)
            r = self._strategy_reward(clones, idx, all_singletons)
            if r is not None:
                print(f"Strategy {idx + 1} → {cname}: {r:.4f}")
                if r > best_reward:
                    best_reward = r
                    best_name = cname

        # 最后考虑新建簇策略
        new_clus = initialize_empty_cluster(name=f"Clus{self._next_id}")
        new_clus.add_bba(name, bba, _init=True)
        clones = [self._clone_cluster(c) for c in self._clusters.values()] + [new_clus]
        r_new = self._strategy_reward(clones, len(clones) - 1, all_singletons)
        if r_new is not None:
            print(f"Strategy {len(clones)} → New Cluster: {r_new:.4f}")
            if r_new > best_reward:
                best_reward = r_new
                best_name = new_clus.name
        return best_name, (best_reward if best_reward >= 0 else None)

    def add_bba_by_reward(self, name: str, bba: BBA) -> str:
        """根据收益自动选择簇并执行入簇，返回目标簇名。"""
        # ---------- 单簇单元边界特判 ---------- #
        if len(self._clusters) == 1:
            only_cluster = next(iter(self._clusters.values()))
            # 如果全局只有一个簇，且该簇只有一个 BBA，则直接判断是否满足分簇阈值，似乎没有更好的办法了。
            if len(only_cluster.get_bbas()) == 1:
                old_bba = only_cluster.get_bbas()[0][1]
                div = bjs_metric(old_bba, bba)
                if div <= THRESHOLD_BJS:
                    only_cluster.add_bba(name, bba)
                    print(f"\nSelected cluster: {only_cluster.name} (by threshold rule)")
                    print()
                    return only_cluster.name
                target_name = f"Clus{self._next_id}"
                new_cluster = initialize_empty_cluster(name=target_name)
                new_cluster.add_bba(name, bba)
                self._clusters[target_name] = new_cluster
                self._next_id += 1
                print(f"\nSelected cluster: {target_name} (by threshold rule)")
                print()
                return target_name

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
