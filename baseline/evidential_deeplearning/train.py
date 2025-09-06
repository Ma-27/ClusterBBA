"""模型训练相关函数
======================

该模块提供了针对 Evidential Deep Learning 的通用训练循环。在保持原有训练逻辑的前提下，加入了更清晰的命令行输出与进度条显示，以便于监控训练过程。
"""

from __future__ import annotations

import copy
import time

import torch
from tqdm.auto import tqdm

from helpers import get_device, one_hot_embedding
from losses import relu_evidence


def train_model(model, dataloaders, num_classes, criterion, optimizer, scheduler=None, num_epochs: int = 25,
                device=None, uncertainty: bool = False, ):
    """训练给定模型并返回最佳模型及统计信息。

    参数
    ----
    model:
        需要训练的 PyTorch 模型。
    dataloaders:
        字典形式的训练与验证集 ``DataLoader``。
    num_classes:
        分类数目，用于构建独热标签与计算不确定性。
    criterion:
        损失函数，支持普通交叉熵或 EDL 损失。
    optimizer:
        优化器实例。
    scheduler:
        可选的学习率调度器。
    num_epochs:
        训练轮数。
    device:
        训练设备，默认为自动检测。
    uncertainty:
        是否计算不确定性指标。
    """

    since = time.time()

    if not device:
        device = get_device()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    losses = {"loss": [], "phase": [], "epoch": []}
    accuracy = {"accuracy": [], "phase": [], "epoch": []}
    evidences = {"evidence": [], "type": [], "epoch": []}

    # 使用 tqdm 展示整体训练进度，避免冗余打印
    # tqdm 进度条实例，命名为 pbar 以便后续更新后缀信息
    pbar = tqdm(range(num_epochs), desc="训练进度", unit="epoch")
    for epoch in pbar:
        epoch_metrics = {}

        # 每个 epoch 分为训练和验证两个阶段
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # 训练模式
            else:
                model.eval()  # 验证模式

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零梯度
                optimizer.zero_grad()

                # 前向传播，仅在训练阶段记录梯度
                with torch.set_grad_enabled(phase == "train"):

                    if uncertainty:
                        y = one_hot_embedding(labels, num_classes).to(device)
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(
                            outputs, y.float(), epoch, num_classes, 10, device
                        )

                        match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
                        acc = torch.mean(match)
                        evidence = relu_evidence(outputs)
                        alpha = evidence + 1
                        u = num_classes / torch.sum(alpha, dim=1, keepdim=True)

                        total_evidence = torch.sum(evidence, 1, keepdim=True)
                        mean_evidence = torch.mean(total_evidence)
                        mean_evidence_succ = torch.sum(
                            torch.sum(evidence, 1, keepdim=True) * match
                        ) / torch.sum(match + 1e-20)
                        mean_evidence_fail = torch.sum(
                            torch.sum(evidence, 1, keepdim=True) * (1 - match)
                        ) / (torch.sum(torch.abs(1 - match)) + 1e-20)

                    else:
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # 统计指标并更新进度条显示当前批次 loss
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if scheduler is not None and phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            losses["loss"].append(epoch_loss)
            losses["phase"].append(phase)
            losses["epoch"].append(epoch)
            accuracy["accuracy"].append(epoch_acc.item())
            accuracy["epoch"].append(epoch)
            accuracy["phase"].append(phase)

            epoch_metrics[f"{phase}_loss"] = f"{epoch_loss:.4f}"
            epoch_metrics[f"{phase}_acc"] = f"{epoch_acc:.4f}"

            # 保存验证阶段表现最佳的模型参数
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # 更新进度条后缀，展示当前 epoch 的损失与准确率
        pbar.set_postfix(epoch_metrics)

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    # 输出最佳验证准确率，保留四位小数
    print(f"Best val Acc: {best_acc:.4f}")

    # 加载最佳模型权重并返回
    model.load_state_dict(best_model_wts)
    metrics = (losses, accuracy)

    return model, metrics
