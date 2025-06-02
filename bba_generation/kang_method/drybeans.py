# -*- coding: utf-8 -*-
import itertools
import math
import os
import random
import zipfile
from collections import defaultdict
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def download_dry_beans(csv_output_path='Dry_Beans_Dataset.csv'):
    # 如果已有CSV文件，直接读取返回
    if os.path.exists(csv_output_path):
        print(f'检测到已有文件：{csv_output_path}，将直接加载...')
        return pd.read_csv(csv_output_path)

    # 下载 Dry Beans 数据集的 zip 文件
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00602/DryBeanDataset.zip'
    print('开始下载 Dry Beans 数据集...')
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"下载失败，状态码：{response.status_code}")
    zip_data = BytesIO(response.content)

    # 解压并读取 Excel 文件
    with zipfile.ZipFile(zip_data, 'r') as zip_ref:
        zip_ref.extractall('DryBeans')
        xlsx_file = [f for f in zip_ref.namelist() if f.endswith('.xlsx')][0]
        print(f"成功解压：{xlsx_file}")

    # 读取 Excel 并保存为 CSV
    df = pd.read_excel(os.path.join('../DryBeans', xlsx_file))
    df.to_csv(csv_output_path, index=False)
    print(f"CSV 文件已保存为：{csv_output_path}")
    return df


# ========== 调试函数区 ========== #
def show_dataset(X, y, df):
    """显示数据集的前几行，带类别标签"""
    df_show = pd.DataFrame(X, columns=df.columns[:-1])
    df_show['target'] = y
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print("示例前五行数据:")
    print(df_show.head())
    print("数据维度:", X.shape)


def show_train_test(X_train, y_train, X_test, y_test, df):
    feature_names = df.columns[:-1]
    #
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train['target'] = y_train
    print("\n训练集（共 %d 个样本）:" % len(df_train))
    print(df_train.head(5))
    print("...")
    print(df_train.tail(5))
    #
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test['target'] = y_test
    print("\n测试集（共 %d 个样本）:" % len(df_test))
    print(df_test.head(5))
    print("...")
    print(df_test.tail(5))


def print_bpa_sum(bpa, df):
    """打印 BPA 的各项值与总和，用于验证归一化"""
    total = 0.0
    for k, v in bpa.items():
        total += v
    print(f"  概率和 Σ = {total:.4f}")


def test_interval_generation(model, df):
    """打印 Dry Beans 区间模型"""
    feature_names = df.columns[:-1]
    class_labels = sorted(df.iloc[:, -1].unique())

    print("\n=== 单类区间数 ===")
    for cls in model.classes:
        key = (cls,)
        label_name = class_labels[cls]
        print(f"\n{label_name} ->")
        print(pd.DataFrame(model.intervals[key], columns=['min', 'max'],
                           index=feature_names))
    print("\n=== 组合类交叉区间 ===")
    for hyp in model.intervals:
        if len(hyp) > 1:
            names = [class_labels[i] for i in hyp]
            print(f"\n{names} ->")
            print(pd.DataFrame(model.intervals[hyp], columns=['min', 'max'],
                               index=feature_names))


def test_single_bpa(model, sample, df):
    """打印 Dry Beans 单样本每个特征的 BPA 分布"""
    feature_names = df.columns[:-1]
    class_labels = sorted(df.iloc[:, -1].unique())
    print("\n=== 单样本各属性的BPA（退化区间） ===")
    for i in range(len(sample)):
        bpa = model.attribute_bpa(sample, i)
        print(f"\n属性 {feature_names[i]}:")
        for k, v in bpa.items():
            labels = ','.join(class_labels[j] for j in k)
            print(f"  {labels:<20} : {v:.4f}")
        # 验证mass function和是不是为1
        print_bpa_sum(bpa, df)


# -------------------------- 工具函数 -------------------------- #

def interval_distance(a, b):
    """
    计算区间数 A=[a1,a2], B=[b1,b2] 的距离 D(A,B)
    公式来源：康兵义等(2012) 定义5，简化积分结果
    D^2 = ((a1-b1)^2 + (a1-b1)*(a2-b2) + (a2-b2)^2)/3
    """
    a1, a2 = a
    b1, b2 = b
    # 等价于论文(19)中的简化结果
    d2 = ((a1 + a2) / 2.0 - (b1 + b2) / 2.0) ** 2 + ((a2 - a1) + (b2 - b1)) ** 2 / 12.0
    return math.sqrt(max(d2, 0.0))


def similarity(a, b, alpha=5.0):
    """
    相似度 s = 1 / (1 + α·D)
    当 b 为“空区间”时（b[0]>b[1]），返回 0
    """
    if b[0] > b[1]:
        return 0.0
    return 1.0 / (1.0 + alpha * interval_distance(a, b))


def normalize(d):
    """将字典值归一化，使其和为1，生成 BPA"""
    total = sum(d.values())
    if total <= 0:
        # 出现极端情况时均匀分给所有假设
        n = len(d)
        return {k: 1.0 / n for k in d}
    return {k: (v / total) for k, v in d.items()}


def dempster_combine(m1, m2):
    """
    Dempster 组合规则（忽略空集质量）：
    m(A) = (Σ_{B∩C=A} m1(B)m2(C)) / (1 - K)
    K = Σ_{B∩C=∅} m1(B)m2(C)
    """
    combined = defaultdict(float)
    conflict = 0.0
    for B, mB in m1.items():
        for C, mC in m2.items():
            inter = tuple(sorted(set(B) & set(C)))
            if not inter:
                conflict += mB * mC
            else:
                combined[inter] += mB * mC
    if conflict >= 1.0:
        raise ValueError("完全冲突，无法组合")
    factor = 1.0 / (1.0 - conflict)
    return {A: m * factor for A, m in combined.items()}


# -------------------------- BPA 生成核心 -------------------------- #

class IntervalBPA:
    """
    根据训练样本构造区间数模型 -> 生成测试样本单属性 BPA
    """

    def __init__(self, X_train, y_train, alpha=5.0, seed=42):
        self.alpha = alpha
        self.rng = random.Random(seed)
        # 类别标签（0,1,...）
        self.classes = sorted(set(y_train))
        # singleton 区间:  {类: ndarray(feature_dim×2)}
        self.intervals = self._build_singleton_intervals(X_train, y_train)
        # 组合事件区间（两两、三类……交集）
        self.intervals.update(self._build_combined_intervals())

    def _build_singleton_intervals(self, X, y):
        intervals = {}
        for c in self.classes:
            cls_samples = X[y == c]
            # 每个特征的 [min, max]
            intervals[(c,)] = np.column_stack((cls_samples.min(0),
                                               cls_samples.max(0)))  # shape = (n_features,2)
        return intervals

    def _build_combined_intervals(self):
        """
        构造多类别组合假设的区间交集。
        仅保留所有维度 low<=high 的组合，否则该假设下 bpa=0
        """
        comb_intervals = {}
        for r in range(2, len(self.classes) + 1):
            for subset in itertools.combinations(self.classes, r):
                mats = [self.intervals[(c,)] for c in subset]
                low = np.maximum.reduce([m[:, 0] for m in mats])
                high = np.minimum.reduce([m[:, 1] for m in mats])
                comb_intervals[subset] = np.column_stack((low, high))
        return comb_intervals

    def attribute_bpa(self, x, attr_idx):
        """
        针对单个属性 idx 生成 BPA，x 是测试样本特征向量
        """
        sim_dict = {}
        for hyp, mat in self.intervals.items():
            a_interval = [x[attr_idx], x[attr_idx]]  # 退化区间
            b_interval = mat[attr_idx]
            sim_dict[hyp] = similarity(a_interval, b_interval, self.alpha)
        return normalize(sim_dict)


# -------------------------- 主流程（Dry Beans） -------------------------- #

def experiment(csv_path='Dry_Beans_Dataset.csv',
               random_state=7,
               alpha=5.0,
               train_ratio=0.4):
    """
    1. 读取 Dry Beans 数据集（CSV 最后一列为 Class）
    2. 标签编码、按类随机划分（train_ratio 比例用于训练）
    3. 构造 IntervalBPA、生成单属性 BPA 并 Dempster 组合
    4. 预测与评估
    返回：accuracy, y_true, y_pred
    """
    # 1. 读取数据
    df = pd.read_csv(csv_path)
    # 丢弃缺失值（如有）
    df = df.dropna().reset_index(drop=True)
    # 特征和标签
    X = df.iloc[:, :-1].values
    raw_labels = df.iloc[:, -1].values
    # 2. 标签编码
    le = LabelEncoder()
    y = le.fit_transform(raw_labels)

    # 3. 随机分层抽样
    rs = np.random.RandomState(random_state)
    train_mask = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        n_train = int(len(idx) * train_ratio)
        chosen = rs.choice(idx, n_train, replace=False)
        train_mask[chosen] = True
    test_mask = ~train_mask
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # 测试和调试函数调用
    show_dataset(X, le.inverse_transform(y), df)
    show_train_test(X_train, le.inverse_transform(y_train), X_test, le.inverse_transform(y_test), df)

    # 4. 构造区间模型
    model = IntervalBPA(X_train, y_train, alpha=alpha, seed=random_state)

    # test_interval_generation(model, df)
    # test_single_bpa(model, X_test[0], df)

    # 5. 针对每个测试样本：单属性 BPA -> Dempster 组合 -> 预测
    y_pred = []
    for x in tqdm(X_test, desc="Dempster 组合", ncols=80):
        # 5.1 各属性 BPA 列表
        attr_bpas = [model.attribute_bpa(x, k) for k in range(X.shape[1])]
        # 5.2 Dempster 级联组合
        fused = attr_bpas[0]
        for bpa in attr_bpas[1:]:
            fused = dempster_combine(fused, bpa)
        # 5.3 按单一类别质量最大化预测
        single_masses = {cls: fused.get((cls,), 0.0) for cls in model.classes}
        pred = max(single_masses, key=single_masses.get)
        y_pred.append(pred)

    # 6. 评估
    acc = accuracy_score(y_test, y_pred)
    print(f'Dry Beans 数据集分类准确率: {acc:.4f}')
    return acc, y_test, y_pred


if __name__ == '__main__':
    df = download_dry_beans()
    print('前5行数据如下：')
    print(df.head())

    acc, y_true, y_hat = experiment(
        csv_path='../Dry_Beans_Dataset.csv',
        random_state=42,
        alpha=5.0,
        train_ratio=0.4
    )
