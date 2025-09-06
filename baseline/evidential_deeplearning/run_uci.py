"""在 UCI 数据集上演示 EDL 训练流程的脚本。"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_openml, load_iris, load_wine
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from helpers import get_device
from losses import edl_mse_loss
from train import train_model

pd.options.display.float_format = "{:.4f}".format


class TabularNet(nn.Module):
    """简单的多层感知机，用于表格数据分类。"""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_dataset(name: str):
    """根据名称加载 UCI 数据集并返回类别名称。"""

    name = name.lower()
    if name == "iris":
        data = load_iris()
        X, y = data.data, data.target
        class_names = list(data.target_names)
    elif name == "wine":
        data = load_wine()
        X, y = data.data, data.target
        class_names = list(data.target_names)
    elif name == "seeds":
        data = fetch_openml(name="seeds", version=1, as_frame=False)
        X = data.data.astype(np.float32)
        y = data.target.astype(int) - 1  # 原始标签从 1 开始
        class_names = [f"class_{i}" for i in np.unique(y)]
    elif name == "glass":
        data = fetch_openml(name="glass", version=1, as_frame=False)
        X = data.data.astype(np.float32)
        le = LabelEncoder()
        y = le.fit_transform(data.target)
        class_names = list(le.classes_)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return X, y, class_names


def prepare_dataloaders(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, batch_size: int = 32, ):
    """划分数据集并构建 DataLoader。"""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    dataloaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val": DataLoader(test_ds, batch_size=batch_size),
    }
    return dataloaders, DataLoader(test_ds, batch_size=batch_size)


def evaluate(model: nn.Module, dataloader: DataLoader, class_names):
    """在测试集上评估模型性能并生成汇总表。"""

    device = get_device()
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    class_f1 = [round(report_dict[name]["f1-score"], 4) for name in class_names]
    summary = pd.DataFrame(
        {
            "Class / Metric": class_names + ["Accuracy", "F1score"],
            "Proposed": class_f1 + [round(acc, 4), round(f1, 4)],
        }
    )

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    return report, round(acc, 4), round(f1, 4), summary, cm_df


def run_experiment(name: str, epochs: int = 50):
    """完整执行一次训练并返回评估结果。"""

    X, y, class_names = load_dataset(name)
    num_classes = len(np.unique(y))
    dataloaders, test_loader = prepare_dataloaders(X, y)
    model = TabularNet(X.shape[1], num_classes)
    device = get_device()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    model, _ = train_model(
        model,
        dataloaders,
        num_classes,
        edl_mse_loss,
        optimizer,
        num_epochs=epochs,
        device=device,
        uncertainty=True,
    )

    report, acc, f1, summary, cm_df = evaluate(model, test_loader, class_names)
    return report, acc, f1, summary, cm_df


if __name__ == "__main__":
    datasets = ["iris", "wine", "seeds", "glass"]
    for ds in datasets:
        print(f"=============== Evidential Deep Learning on {ds} Dataset ===============")
        rep, acc, f1, summary, cm_df = run_experiment(ds)
        print(rep)
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 score: {f1:.4f}")
        print(summary.to_markdown(index=False, floatfmt=".4f"))
        print("\nConfusion Matrix:")
        print(cm_df.to_string())
        print()
