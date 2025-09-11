"""DS+SVM 基线的经典数据集读取工具。"""
from typing import List, Tuple

import numpy as np
from sklearn.datasets import fetch_openml, load_iris, load_wine
from sklearn.preprocessing import LabelEncoder


def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """读取一个经典的 UCI 数据集。

    参数
    ----
    name: str
        数据集名称（iris、wine、seeds 或 glass）。
    返回
    ----
    Tuple[np.ndarray, np.ndarray, List[str]]
        特征矩阵 ``X``、标签 ``y``（从 1 开始编码）以及类别名称列表。
    """
    # 名称统一为小写，方便与支持的关键字进行匹配
    name = name.lower()
    print(f"[INFO] 正在加载数据集：{name}…")
    # 直接调用 scikit-learn 自带的加载函数
    if name == "iris":
        # Iris 鸢尾花数据集，scikit-learn 内置
        d = load_iris()
        # 特征矩阵 ``d.data``，标签 ``d.target`` 需加一转换为 1..K
        X, y = d.data, d.target + 1
        class_names = list(d.target_names)
    elif name == "wine":
        # Wine 葡萄酒数据集，scikit-learn 内置
        d = load_wine()
        X, y = d.data, d.target + 1
        class_names = list(d.target_names)
    # seeds 与 glass 需要从 OpenML 获取
    elif name == "seeds":
        # Seeds 谷物数据集，从 OpenML 下载
        d = fetch_openml(name="seeds", version=1, as_frame=False)
        # OpenML 返回的 ``data`` 与 ``target`` 需转为浮点与整数类型
        X = d.data.astype(np.float64)
        y = d.target.astype(int)
        # 该数据集中类别名称缺失，故构造通用名称
        class_names = [f"class_{i}" for i in sorted(np.unique(y))]
    elif name == "glass":
        # Glass 玻璃种类数据集，同样从 OpenML 下载
        d = fetch_openml(name="glass", version=1, as_frame=False)
        X = d.data.astype(np.float64)
        le = LabelEncoder()
        # 将原始字符标签转为 1..K 的整数编码
        y = le.fit_transform(d.target) + 1
        class_names = list(le.classes_)
    else:
        # 若名称不在支持列表中则抛出异常
        raise ValueError(f"Unknown dataset: {name}")

    print(f"[INFO] 数据集 {name} 加载完成，样本数 = {X.shape[0]}")
    return X, y, class_names
