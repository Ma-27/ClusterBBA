"""二分类证据化 Logistic 标定（原方法 A）。"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """稳定的 sigmoid 函数实现。"""
    # 避免指数溢出，先把输入截断在 [-50, 50]
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class BinaryEvidentialLogisticCalibration:
    """针对一对余 SVM 得分的证据化 Logistic 标定。"""

    # 训练/校准集的分数与标签
    s_cal: np.ndarray
    y_cal: np.ndarray

    # 数值积分与网格配置
    n_theta1: int = 121
    theta1_std_range: float = 4.0
    n_tau: int = 501

    def _platt_targets(self) -> Tuple[float, float]:
        """生成 Platt 平滑的目标值以防过拟合。"""
        # 正负样本数量
        n_pos = int(np.sum(self.y_cal == 1))
        n_neg = int(np.sum(self.y_cal == 0))
        # 根据 Platt 提出的公式得到平滑后的概率目标
        t_pos = (n_pos + 1.0) / (n_pos + 2.0)
        t_neg = 1.0 / (n_neg + 2.0)
        return t_pos, t_neg

    def _log_likelihood(self, theta0: float, theta1: float) -> float:
        """给定参数计算对数似然。"""
        t_pos, t_neg = self._platt_targets()
        z = theta0 + theta1 * self.s_cal
        p = sigmoid(z)
        # 根据样本标签选择对应的平滑目标
        t = np.where(self.y_cal == 1, t_pos, t_neg)
        eps = 1e-12
        # 返回交叉熵的负值作为对数似然
        return float(np.sum(t * np.log(p + eps) + (1.0 - t) * np.log(1.0 - p + eps)))

    def _fit_mle(self) -> Tuple[float, float, np.ndarray]:
        """用牛顿法求解 (θ0, θ1) 的极大似然估计并给出协方差。"""
        theta = np.zeros(2, dtype=float)
        for _ in range(200):
            z = theta[0] + theta[1] * self.s_cal
            p = sigmoid(z)
            t_pos, t_neg = self._platt_targets()
            t = np.where(self.y_cal == 1, t_pos, t_neg)
            # 梯度向量
            g0 = np.sum(t - p)
            g1 = np.sum((t - p) * self.s_cal)
            # Hessian 矩阵
            w = p * (1.0 - p)
            H00 = np.sum(w)
            H01 = np.sum(w * self.s_cal)
            H11 = np.sum(w * (self.s_cal ** 2))
            H = np.array([[H00, H01], [H01, H11]]) + 1e-8 * np.eye(2)
            step = np.linalg.solve(H, np.array([g0, g1]))
            # 线搜索控制步长，保证似然递增
            alpha = 1.0
            prev_ll = self._log_likelihood(theta[0], theta[1])
            for _ in range(10):
                trial = theta + alpha * step
                ll = self._log_likelihood(trial[0], trial[1])
                if ll >= prev_ll - 1e-8:
                    theta = trial
                    break
                alpha *= 0.5
            if np.linalg.norm(alpha * step) < 1e-6:
                break
        # 由 Hessian 的逆得到参数协方差矩阵
        z = theta[0] + theta[1] * self.s_cal
        p = sigmoid(z)
        w = p * (1.0 - p)
        H00 = np.sum(w)
        H01 = np.sum(w * self.s_cal)
        H11 = np.sum(w * (self.s_cal ** 2))
        H = np.array([[H00, H01], [H01, H11]]) + 1e-8 * np.eye(2)
        cov = np.linalg.inv(H)
        return theta[0], theta[1], cov

    def _pl_theta(self, theta0: float, theta1: float, ll_mle: float) -> float:
        """计算给定参数点的相对似然。"""
        return float(np.exp(self._log_likelihood(theta0, theta1) - ll_mle))

    def plausibility_positive(self, s_test: float) -> float:
        """对测试分数计算其属于正类的可能性 (plausibility)。"""
        # 先拟合参数并获取协方差，用于构造 θ1 的扫描范围
        theta0_hat, theta1_hat, cov = self._fit_mle()
        ll_hat = self._log_likelihood(theta0_hat, theta1_hat)
        var1 = max(cov[1, 1], 1e-6)
        std1 = np.sqrt(var1)
        theta1_grid = np.linspace(
            theta1_hat - self.theta1_std_range * std1,
            theta1_hat + self.theta1_std_range * std1,
            self.n_theta1,
        )
        # 对 τ∈(0,1) 的网格逐点计算 pl_T(τ)
        tau_grid = np.linspace(1e-4, 1.0 - 1e-4, self.n_tau)
        pl_tau = np.zeros_like(tau_grid)
        for i, tau in enumerate(tau_grid):
            logit_tau = np.log(tau / (1.0 - tau))
            # θ0 = -logit_tau - θ1 * s_test
            theta0_line = (-logit_tau) - theta1_grid * s_test
            vals = [self._pl_theta(t0, t1, ll_hat) for t0, t1 in zip(theta0_line, theta1_grid)]
            pl_tau[i] = np.max(vals)
        # 根据论文公式，最终的可能性值等于 τ_hat 加上积分部分
        tau_hat = float(sigmoid(theta0_hat + theta1_hat * s_test))
        pl_val = tau_hat
        right = tau_grid[tau_grid >= tau_hat]
        if right.size > 1:
            pr = pl_tau[tau_grid >= tau_hat]
            pl_val += np.trapz(pr, right)
        return float(np.clip(pl_val, 0.0, 1.0))
