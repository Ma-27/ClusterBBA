# -*- coding: utf-8 -*-
"""多簇管理模块
===============

提供 ``MultiClusters`` 类以统一管理多个 ``Cluster`` 对象，并根据
BBA 的收益策略完成自动分簇流程。
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

# 依赖本项目内现成工具函数 / 模块
from cluster.one_cluster import Cluster, initialize_empty_cluster  # type: ignore
from config import THRESHOLD_BJS, SPLIT_TIMES, INTRA_EPS, DP_PENALTY
from divergence.bjs import bjs_metric
from divergence.rd_ccjs import divergence_matrix  # type: ignore
from mean.mean_divergence import average_divergence  # type: ignore
from utility.bba import BBA

__all__ = [
    "MultiClusters",
    "construct_clusters_by_sequence",
    "construct_clusters_by_sequence_dp",
]


class MultiClusters:
    """维护整个簇集合 `C` 的高层数据结构。

    参数 ``debug`` 控制是否在执行过程中打印调试信息。
    """

    def __init__(self, debug: bool = True) -> None:
        self._clusters: Dict[str, Cluster] = {}
        self._next_id: int = 1  # 自动生成新簇名称使用
        self._debug: bool = debug

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
        for b in src.get_bbas():
            clone.add_bba(b, _init=True)
        return clone

    @staticmethod
    def _calc_intra_divergence(target: Cluster, clusters: List[Cluster], handle_boundary: bool = False) -> Optional[
        float]:
        """按照定义计算 ``D_intra``。

        ``handle_boundary`` 为 ``True`` 时，保持旧版边界处理逻辑：
        当目标簇仅含一个 BBA 时，使用其他簇的平均 ``D_intra`` 进行替代；否则在无法计算时返回 ``None``。当 ``handle_boundary`` 为 ``False``（默认）时，只要目标簇不足两条 BBA，直接返回 ``None``。
        """

        # 若簇规模达到 2，则直接计算其内部散度
        if len(target.get_bbas()) >= 2:
            return target.intra_divergence()

        # 不处理边界条件时，直接返回 None
        if not handle_boundary:
            return None

        # ---------- 旧版边界特殊处理 ---------- #
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
            for b in left:
                clus_l.add_bba(b, _init=True)
            for b in right:
                clus_r.add_bba(b, _init=True)
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
                di = MultiClusters._calc_intra_divergence(clus, clusters, handle_boundary=True)
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
    def _evaluate_strategies(self, bba: BBA) -> Tuple[str, Optional[float]]:
        """内部使用，评估所有策略并返回最优簇名称。"""
        cluster_names = list(self._clusters.keys())
        best_reward: float = -1.0
        best_name = ""
        if self._debug:
            print("Rewards of strategies:")

        # 判断是否处于多簇全为单元簇的边界情形
        all_singletons = len(self._clusters) >= 2 and all(
            len(c.get_bbas()) == 1 for c in self._clusters.values()
        )

        # 遍历现有簇，将 BBA 加入其中并计算各自的收益
        for idx, cname in enumerate(cluster_names):
            # 获取当前簇
            clones = [self._clone_cluster(c) for c in self._clusters.values()]
            clones[idx].add_bba(bba, _init=True)
            r = self._strategy_reward(clones, idx, all_singletons)
            if r is not None:
                if self._debug:
                    print(f"Strategy {idx + 1} → {cname}: {r:.4f}")
                if r > best_reward:
                    best_reward = r
                    best_name = cname

        # 最后考虑新建簇策略
        new_clus = initialize_empty_cluster(name=f"Clus{self._next_id}")
        new_clus.add_bba(bba, _init=True)
        clones = [self._clone_cluster(c) for c in self._clusters.values()] + [new_clus]
        r_new = self._strategy_reward(clones, len(clones) - 1, all_singletons)
        if r_new is not None:
            if self._debug:
                print(f"Strategy {len(clones)} → New Cluster: {r_new:.4f}")
            if r_new > best_reward:
                best_reward = r_new
                best_name = new_clus.name
        return best_name, (best_reward if best_reward >= 0 else None)

    def add_bba_by_reward(self, bba: BBA) -> str:
        """根据收益自动选择簇并执行入簇，返回目标簇名。"""
        # ---------- 单簇单元边界特判 ---------- #
        if len(self._clusters) == 1:
            only_cluster = next(iter(self._clusters.values()))
            # 如果全局只有一个簇，且该簇只有一个 BBA，则直接判断是否满足分簇阈值，似乎没有更好的办法了。
            if len(only_cluster.get_bbas()) == 1:
                old_bba = only_cluster.get_bbas()[0]
                div = bjs_metric(old_bba, bba)
                if div <= THRESHOLD_BJS:
                    only_cluster.add_bba(bba)
                    if self._debug:
                        print(f"\nSelected cluster: {only_cluster.name} (by threshold rule)")
                        print()
                    return only_cluster.name
                target_name = f"Clus{self._next_id}"
                new_cluster = initialize_empty_cluster(name=target_name)
                new_cluster.add_bba(bba)
                self._clusters[target_name] = new_cluster
                self._next_id += 1
                if self._debug:
                    print(f"\nSelected cluster: {target_name} (by threshold rule)")
                    print()
                return target_name

        target_name, reward = self._evaluate_strategies(bba)
        # 无法评价时，新建簇，通常对应没有任何簇的情况
        if not target_name:
            target_name = f"Clus{self._next_id}"
            if self._debug:
                print("No one cluster found, no reward yet...")
            reward = None
        # 如果目标簇不存在，则说明初始化一个空簇并添加 BBA
        if target_name not in self._clusters:
            clus = initialize_empty_cluster(name=target_name)
            self._clusters[target_name] = clus
            self._next_id += 1
        clus = self._clusters[target_name]
        clus.add_bba(bba)
        if self._debug:
            print(
                f"\nSelected cluster: {target_name}, reward={reward:.4f}" if reward is not None else f"Selected cluster: {target_name}")
            print()
        return target_name


def construct_clusters_by_sequence(bbas: List[BBA], debug: bool = False) -> "MultiClusters":
    """按照给定顺序批量加入 BBA 并返回 :class:`MultiClusters` 对象。

    ``debug`` 为 ``False`` 时不会打印分簇过程信息。
    """
    mc = MultiClusters(debug=debug)
    for bba in bbas:
        mc.add_bba_by_reward(bba)
    return mc


def construct_clusters_by_sequence_dp(bbas: List[BBA], debug: bool = False) -> "MultiClusters":
    """使用动态规划寻找 ``D_intra`` 最小的全局簇划分。

    为了避免分簇结果受 BBA 加入顺序影响，函数内部会先按名称
    对 ``bbas`` 进行排序，再在此固定顺序上执行区间动态规划。
    代价函数在 ``D_intra`` 的基础上引入 ``DP_PENALTY``，
    用于抑制所有 BBA 单独成簇的退化解。

    ``debug`` 为 ``False`` 时不输出任何信息。
    """

    # 统一按照 BBA 名称排序，保证输入顺序的一致性
    # 根据名称中的数字顺序排序，如 m1, m2, m10 ...
    def _name_key(b: BBA) -> int:
        name = b.name.lstrip("mM")
        return int(name) if name.isdigit() else 0

    bbas = sorted(bbas, key=_name_key)

    def _cluster_cost(named: List[BBA]) -> float:
        clus = initialize_empty_cluster(name="tmp")
        for b in named:
            clus.add_bba(b, _init=True)
        di = clus.intra_divergence() or 0.0
        # 加入惩罚项，避免出现过多簇
        return di + DP_PENALTY

    n = len(bbas)
    dp: List[Tuple[float, List[List[BBA]]]] = [(float("inf"), []) for _ in range(n + 1)]
    dp[0] = (0.0, [])

    for i in range(1, n + 1):
        best_val = float("inf")
        best_part: List[List[BBA]] = []
        for j in range(0, i):
            cost = _cluster_cost(bbas[j:i]) + dp[j][0]
            if cost < best_val:
                best_val = cost
                best_part = dp[j][1] + [bbas[j:i]]
        dp[i] = (best_val, best_part)

    mc = MultiClusters(debug=debug)
    for idx, clus_bbas in enumerate(dp[n][1], start=1):
        clus = initialize_empty_cluster(name=f"Clus{idx}")
        for b in clus_bbas:
            clus.add_bba(b, _init=True)
        mc.add_cluster(clus)
    return mc
