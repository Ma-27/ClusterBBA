"""EDL 损失函数与相关计算。

包含论文中常见的 KL 散度、对数似然损失等实现。
"""

import torch
import torch.nn.functional as F

from helpers import get_device


def relu_evidence(y: torch.Tensor) -> torch.Tensor:
    """对网络输出取 ReLU 以获得非负证据。"""

    return F.relu(y)


def kl_divergence(alpha: torch.Tensor, num_classes: int, device=None) -> torch.Tensor:
    """Dirichlet 分布的 KL 散度项。"""

    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y: torch.Tensor, alpha: torch.Tensor, device=None) -> torch.Tensor:
    """计算 EDL 中的对数似然损失。"""

    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y: torch.Tensor, alpha: torch.Tensor, epoch_num: int, num_classes: int, annealing_step: int,
             device=None, ) -> torch.Tensor:
    """均方误差形式的 EDL 损失。"""

    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_mse_loss(output: torch.Tensor, target: torch.Tensor, epoch_num: int, num_classes: int, annealing_step: int,
                 device=None, ) -> torch.Tensor:
    """对外暴露的 MSE 形式 EDL 损失封装。"""

    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss
