# -*- coding: utf-8 -*-
"""多簇管理模块
===============

提供 ``MultiClusters`` 类以统一管理多个 ``Cluster`` 对象，并根据 BBA 的收益策略完成自动分簇流程。
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

# 依赖本项目内现成工具函数 / 模块
from cluster.one_cluster import Cluster, initialize_empty_cluster  # type: ignore
from config import (
    threshold_bjs,
    SPLIT_TIMES,
    INTRA_EPS,
    LAMBDA as DEFAULT_LAMBDA,
    MU as DEFAULT_MU,
)
from divergence.bjs import bjs_metric
from divergence.rd_ccjs import divergence_matrix  # type: ignore
from mean.mean_divergence import average_divergence  # type: ignore
from utility.bba import BBA

__all__ = [
    "MultiClusters",
    "construct_clusters_by_sequence",
    "construct_clusters_by_sequence_dp",
]

# 默认超参, 提供给外部可选参数
LAMBDA: float = DEFAULT_LAMBDA
MU: float = DEFAULT_MU


class MultiClusters:
    """维护整个簇集合 `C` 的高层数据结构。

    参数 ``debug`` 控制是否在执行过程中打印调试信息。
    """

    def __init__(self, debug: bool = True, *, lambda_val: float | None = None, mu_val: float | None = None, ) -> None:
        self._clusters: Dict[str, Cluster] = {}
        self._next_id: int = 1  # 自动生成新簇名称使用
        self._debug: bool = debug
        self.lambda_val = LAMBDA if lambda_val is None else lambda_val
        self.mu_val = MU if mu_val is None else mu_val

    # ------------------------------------------------------------------
    # 基础操作
    # ------------------------------------------------------------------
    def add_cluster(self, cluster: Cluster) -> None:
        """注册一个新的簇。"""
        if cluster.name in self._clusters:
            raise ValueError(f"Duplicate cluster name: {cluster.name}")
        self._clusters[cluster.name] = cluster
        # 新增簇时同步更新 ``_next_id``，以免后续自动命名发生冲突
        if cluster.name.startswith("Clus"):
            try:
                idx = int(cluster.name[4:])
                self._next_id = max(self._next_id, idx + 1)
            except ValueError:
                pass

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
    def _strategy_reward(clusters: List[Cluster], idx: int, *, lambda_val: float, mu_val: float,
                         all_singletons: bool = False, ) -> Optional[float]:
        """计算策略 ``k`` 的收益 ``R_k``。``idx`` 为含新 BBA 的簇索引。

        ``idx`` 为含新 BBA 的簇索引。 ``all_singletons`` 指示是否处于多簇均为单元簇
        的特殊边界情形。收益定义为

        .. math::

           R_k=\frac{\bigl(\tfrac{1}{P_k}\sum RD_{CCJS}(Clus_i,Clus_j)\bigr)^{\mu}}
           {\bigl(\tfrac{1}{K_k}\sum D_{intra}(Clus_i^{+})\bigr)^{\lambda}}

        其中 ``mu_val`` 与 ``lambda_val`` 为灵敏度系数。

        ``all_singletons`` 标记在调用时是否处于多簇全为单元簇的边界情况，如果是的话，则要执行非常特殊的处理。
        """
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
        if avg_d_intra <= INTRA_EPS:
            # 若 ``avg_d_intra`` 极小, 在大 \lambda 情形下可能因浮点下溢导致 0 除,因此将其下界限制为 ``INTRA_EPS``.
            avg_d_intra = INTRA_EPS

        # 计算收益，注意，\mu 与 \lambda 是可调的，但是如果不传参就是默认值。
        # 直接在普通浮点下计算可能出现下溢或溢出，这里使用 ``numpy`` 的 ``longdouble``提供更大的浮点范围。

        base_rd_cc = np.longdouble(max(avg_rd_cc, INTRA_EPS))
        base_d_intra = np.longdouble(max(avg_d_intra, INTRA_EPS))

        numerator = np.power(base_rd_cc, np.longdouble(mu_val))
        denominator = np.power(base_d_intra, np.longdouble(lambda_val))

        if denominator == 0.0:
            return float("inf")

        reward_ld = numerator / denominator
        info = np.finfo(np.longdouble)
        if reward_ld > info.max:
            return float("inf")
        if reward_ld < info.tiny:
            return 0.0
        return float(reward_ld)

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
            r = self._strategy_reward(clones, idx, lambda_val=self.lambda_val, mu_val=self.mu_val,
                                      all_singletons=all_singletons, )
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
        # 新建簇的收益
        r_new = self._strategy_reward(clones, len(clones) - 1, lambda_val=self.lambda_val, mu_val=self.mu_val,
                                      all_singletons=all_singletons, )
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
                # 根据 Frame of Discernment 动态计算每个问题的二分阈值
                thresh = threshold_bjs(bba.theta_size)
                if self._debug:
                    print(
                        f"The divergence between {old_bba.name} and {bba.name} is {div:.4f}. The current threshold is {thresh:.4f}.")
                if div <= thresh:
                    only_cluster.add_bba(bba)
                    return only_cluster.name
                target_name = f"Clus{self._next_id}"
                new_cluster = initialize_empty_cluster(name=target_name)
                new_cluster.add_bba(bba)
                self._clusters[target_name] = new_cluster
                self._next_id += 1
                if self._debug:
                    print(f"Selected cluster: {target_name} (by threshold rule)")
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


def construct_clusters_by_sequence(bbas: List[BBA], debug: bool = False, *, lambda_val: float | None = None,
                                   mu_val: float | None = None, ) -> "MultiClusters":
    """按照给定顺序批量加入 BBA 并返回 :class:`MultiClusters` 对象。

    ``debug`` 为 ``False`` 时不会打印分簇过程信息。
    ``lambda_val`` 与 ``mu_val`` 控制收益计算中的灵敏度系数。
    """
    mc = MultiClusters(debug=debug, lambda_val=lambda_val, mu_val=mu_val)
    for bba in bbas:
        if debug:
            print(f"------------------------------ Round: {bba.name} ------------------------------ ")
        mc.add_bba_by_reward(bba)
    return mc


def construct_clusters_by_sequence_dp(bbas: List[BBA], debug: bool = False, *, lambda_val: float | None = None,
                                      mu_val: float | None = None, ) -> "MultiClusters":
    """在固定的 BBA 顺序上使用动态规划搜索最高收益的簇划分。

    参数 ``bbas`` 的顺序不会被修改, 函数会在此顺序基础上枚举所有可能的区间划分。每一种划分都会计算一次全局收益 ``R_k``，其定义见 :func:`_strategy_reward`，返回收益最高的结果。

    ``debug`` 为 ``False`` 时不输出任何信息。
    ``lambda_val`` 与 ``mu_val`` 控制收益计算中的灵敏度系数。
    """

    def _partition_reward(partition: List[List[BBA]]) -> Optional[float]:
        """给定一个 BBA 划分, 计算其全局收益。"""
        clusters: List[Cluster] = []
        for idx, seg in enumerate(partition, start=1):
            # 将当前区间 ``seg`` 转换为临时簇，便于后续统一计算收益
            clus = initialize_empty_cluster(name=f"Tmp{idx}")
            for b in seg:
                # 枚举区间中的每个 BBA，依次加入临时簇
                clus.add_bba(b, _init=True)
            clusters.append(clus)

        # 判断此划分是否落在多簇全为单元簇的边界情况
        all_singletons = len(clusters) >= 2 and all(len(c.get_bbas()) == 1 for c in clusters)
        # 复用 ``_strategy_reward`` 评价该划分的整体收益
        return MultiClusters._strategy_reward(clusters, 0, lambda_val=lambda_val, mu_val=mu_val,
                                              all_singletons=all_singletons, )

    n = len(bbas)
    # dp[i] = (best_reward, best_partition for bbas[:i])
    # 向量 ``dp`` 记录前 ``i`` 个 BBA 的最优收益及对应划分
    dp: List[Tuple[float, List[List[BBA]]]] = [(-float("inf"), []) for _ in range(n + 1)]
    dp[0] = (0.0, [])

    for i in range(1, n + 1):
        # 尝试确定前 ``i`` 个 BBA 的最优区间划分
        best_val = -float("inf")
        best_part: List[List[BBA]] = []
        for j in range(0, i):
            # 枚举最后一个分割点 ``j``，区间 [j:i] 作为最后一簇
            part = dp[j][1] + [bbas[j:i]]
            r = _partition_reward(part)
            if r is None:
                continue
            if r > best_val:
                # 更新当前最优收益与划分
                best_val = r
                best_part = part
        # 记录前 ``i`` 个 BBA 的最优解
        dp[i] = (best_val, best_part)

    # 根据动态规划得到的最优划分重新构造 ``MultiClusters`` 对象
    mc = MultiClusters(debug=debug, lambda_val=lambda_val, mu_val=mu_val)
    for idx, clus_bbas in enumerate(dp[n][1], start=1):
        clus = initialize_empty_cluster(name=f"Clus{idx}")
        for b in clus_bbas:
            if debug:
                print(f"------------------------------ Round: {b.name} ------------------------------ ")
            # 将对应 BBA 加入新簇
            clus.add_bba(b, _init=True)
        mc.add_cluster(clus)
    return mc
