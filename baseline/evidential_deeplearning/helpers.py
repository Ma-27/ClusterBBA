"""本包的一些辅助函数，包含设备选择与独热编码等常用工具。
"""

import torch


def get_device() -> torch.device:
    """根据环境自动选择 CPU 或 GPU 设备。"""

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


def one_hot_embedding(labels: torch.Tensor, num_classes: int = 10) -> torch.Tensor:
    """将标签张量转换为独热编码形式。"""

    y = torch.eye(num_classes)
    return y[labels]

