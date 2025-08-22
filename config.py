# -*- coding: utf-8 -*-
"""项目全局超参数配置"""

import math

# 对数底数统一为 2
LOG_BASE: float = 2.0

# RD_CCJS 权重平滑参数
SCALE_DELTA: float = 8e-3
SCALE_EPSILON: float = 4e-3

# 计算簇间收益 d_intra 的默认 epsilon，用于处理多簇全为单元簇 (K>=2, forall n_i=1) 的小常数
INTRA_EPS: float = 1e-6

# RD_CCJS 随机二分次数
SPLIT_TIMES: int = 5

# 计算 BJS / D_CCJS 时避免取 log(0) 的极小常数
EPS: float = 1e-12

# 默认分形深度，todo 仅在模块测试时启用，可根据实际情况修改
SEG_DEPTH: int = 2

# 实验中随机打乱顺序的重复次数
SHUFFLE_TIMES: int = 100

# tqdm 进度条宽度
PROGRESS_NCOLS: int = 120

# Information Volume 中的 epsilon
IV_EPSILON: float = 0.001

# 专家超参数 $\alpha$，用于最后的证据融合与证据折扣。
ALPHA: float = 1
# $\mu$ 是簇-簇散度灵敏度系数（小于 1 的话，越小越放大；大于 1 的话，越大越弱化）
MU: float = 1.0
# $\lambda$ 是簇内散度灵敏度系数（小于 1 的话，越小越放大；大于 1 的话，越大越弱化）
LAMBDA: float = 1.0

# 动态规划分簇时对簇数量的惩罚项，防止出现大量单元素簇
DP_PENALTY: float = 0.1

# 交叉验证折数
K_FOLD_SPLITS = 5


# -------------------------------- 函数 ----------------------------------------------

# 单簇单元判断阈值，本质是 BJS 阈值，用于处理单簇单元 (K=1, n1=1) 边界
# 此值可能会影响初始划分结果
def threshold_bjs(theta_size: int) -> float:
    """根据识别框架大小计算 BJS 阈值。

    参数 ``theta_size`` 为识别框架 ``|Θ|`` 的元素个数。公式来自论文中的
    τ(|Θ|) = sqrt(1 - (π/4)·K·[Γ(K)/Γ(K+1/2)]^2)，其中 K = 2^|Θ| - 1。
    """
    k = (2 ** theta_size) - 1
    log_ratio = math.lgamma(k) - math.lgamma(k + 0.5)
    eb = (math.pi / 4.0) * k * math.exp(2.0 * log_ratio)
    return math.sqrt(max(0.0, 1.0 - eb))
