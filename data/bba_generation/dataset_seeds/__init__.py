from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path(__file__).resolve().parent / "seeds_dataset.txt"


def load_seeds_data() -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """加载 Seeds 数据集。

    返回
        X_all: 所有特征数据
        y_all: 标签 (从 0 开始)
        attr_names: 属性名称列表
        class_names: 类别简称列表
        full_class_names: 类别全称列表
    """
    df_raw = pd.read_csv(DATA_FILE, sep="\s+", header=None)
    X_all = df_raw.iloc[:, :-1].to_numpy(dtype=float)
    y_all = df_raw.iloc[:, -1].to_numpy(dtype=int) - 1
    attr_names = ["A", "P", "C", "KL", "KW", "AC", "KG"]
    class_names = ["Ka", "Ro", "Ca"]
    full_class_names = ["Kama", "Rosa", "Canadian"]
    return X_all, y_all, attr_names, class_names, full_class_names
