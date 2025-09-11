"""多类证据化校准（原方法 B）。"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np


def stable_softmax(theta: float, s_vec: np.ndarray) -> np.ndarray:
    """数值稳定的 softmax 计算。"""
    # 先减去最大值避免溢出
    z = theta * s_vec
    z = z - np.max(z)
    ez = np.exp(z)
    return ez / np.sum(ez)


@dataclass
class EvidentialMultinomialCalibration:
    """针对多分类 OVA 得分的证据化多项式校准。"""

    # 校准集的分数矩阵与标签
    S_cal: np.ndarray
    y_cal: np.ndarray

    # 参数 θ 的搜索范围与蒙特卡洛配置
    theta_max: float = 20.0
    n_theta_grid: int = 801
    mc_M: int = 400
    interval_theta_samples: int = 31

    def _class_counts(self, K: int) -> np.ndarray:
        """统计每个类别在校准集中的样本数。"""
        cnt = np.zeros(K, dtype=int)
        for k in range(1, K + 1):
            cnt[k - 1] = np.sum(self.y_cal == k)
        return cnt

    def _neg_log_like_smoothed(self, theta: float, K: int, class_counts: np.ndarray) -> float:
        """带平滑项的负对数似然，用于避免过拟合。"""
        n = self.S_cal.shape[0]
        s = self.S_cal
        total = 0.0
        for i in range(n):
            # 当前样本的真实类别（从 0 开始）
            kstar = int(self.y_cal[i]) - 1
            nk = class_counts[kstar]
            denom = nk + K
            # t_vec 是带平滑的 one-hot 目标
            t_vec = np.full(K, 1.0 / denom, dtype=float)
            t_vec[kstar] = (nk + 1.0) / denom
            p = stable_softmax(theta, s[i, :])
            total -= np.sum(t_vec * np.log(np.clip(p, 1e-12, 1.0)))
        return total

    def _fit_theta_mle(self, K: int) -> float:
        """使用黄金分割搜索求 θ 的极大似然估计。"""
        cnt = self._class_counts(K)

        def fobj(th):
            return self._neg_log_like_smoothed(th, K, cnt)

        a, b = 0.0, self.theta_max
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = fobj(c)
        fd = fobj(d)
        for _ in range(120):
            if fc > fd:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = fobj(d)
            else:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = fobj(c)
            if abs(b - a) < 1e-4:
                break
        th_hat = c if fc < fd else d
        return float(th_hat)

    def _precompute_pl_theta(self, K: int, theta_hat: float) -> Tuple[np.ndarray, np.ndarray]:
        """预先计算参数空间上的可能性曲线。"""
        cnt = self._class_counts(K)
        grid = np.linspace(0.0, self.theta_max, self.n_theta_grid)
        L_hat = -self._neg_log_like_smoothed(theta_hat, K, cnt)
        pl = np.zeros_like(grid)
        for i, th in enumerate(grid):
            L_th = -self._neg_log_like_smoothed(th, K, cnt)
            pl[i] = np.exp(L_th - L_hat)
        pl = np.clip(pl / (np.max(pl) + 1e-12), 0.0, 1.0)
        return grid, pl

    def _interval_from_gamma(self, grid: np.ndarray, pl: np.ndarray, gamma: float) -> Tuple[float, float]:
        """根据 γ 从可能性曲线中得到 θ 的区间。"""
        mask = pl >= gamma
        idx = np.where(mask)[0]
        if idx.size == 0:
            j = int(np.argmax(pl))
            return float(grid[j]), float(grid[j])
        return float(grid[idx[0]]), float(grid[idx[-1]])

    def pl_singleton(self, s_test: np.ndarray) -> np.ndarray:
        """对单个测试样本计算各类别的等高函数值。"""
        K = s_test.shape[0]
        theta_hat = self._fit_theta_mle(K)
        grid, pl = self._precompute_pl_theta(K, theta_hat)
        rng = np.random.default_rng(12345)
        pl_count = np.zeros(K, dtype=float)
        # 预先生成随机数以加速循环
        gammas = rng.random(self.mc_M)
        Zs = rng.random(self.mc_M)
        for i in range(self.mc_M):
            gamma = gammas[i]
            Z = Zs[i]
            th_lo, th_hi = self._interval_from_gamma(grid, pl, gamma)
            if th_hi < th_lo:
                th_lo, th_hi = th_hi, th_lo
            # 当区间极窄时只取一个点
            if abs(th_hi - th_lo) < 1e-9:
                thetas = np.array([th_lo])
            else:
                thetas = np.linspace(th_lo, th_hi, self.interval_theta_samples)
            possible = set()
            for th in thetas:
                p = stable_softmax(th, s_test)
                cdf = np.cumsum(p)
                j = int(np.searchsorted(cdf, Z, side="right"))
                j = min(j, K - 1)
                possible.add(j)
            for j in possible:
                pl_count[j] += 1.0
        return pl_count / float(self.mc_M)
