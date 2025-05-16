import itertools
import math
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from tqdm import tqdm


# -------------------------- 工具函数 -------------------------- #
def interval_distance(a, b):
    """
    计算区间数 A=[a1,a2], B=[b1,b2] 的距离 D(A,B)
    公式来源：康兵义等(2012) 定义5，简化积分结果
    D^2 = ((a1-b1)^2 + (a1-b1)*(a2-b2) + (a2-b2)^2)/3
    """
    a1, a2 = a
    b1, b2 = b
    # d2 = ((a1 - b1) ** 2 + (a1 - b1) * (a2 - b2) + (a2 - b2) ** 2) / 3.0
    d2 = ((a1 + a2) / 2.0 - (b1 + b2) / 2.0) ** 2 + ((a2 - a1) + (b2 - b1)) ** 2 / 12.0
    return math.sqrt(max(d2, 0.0))


def similarity(a, b, alpha=5.0):
    """
    当 b 为“空区间”时（b[0]>b[1]），返回 0
    """
    if b[0] > b[1]:
        return 0.0
    """相似度 s = 1 / (1 + α·D)"""
    return 1.0 / (1.0 + alpha * interval_distance(a, b))


def normalize(d):
    """将字典值归一化，使其和为1，生成BPA"""
    total = sum(d.values())
    return {k: (v / total if total > 0 else 0.0) for k, v in d.items()}


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
    if conflict == 1.0:
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

        # 类别标签
        self.classes = sorted(set(y_train))
        # singleton 区间:  {类: [min,max]}
        self.intervals = self._build_singleton_intervals(X_train, y_train)
        # 组合事件区间（两两 + 三类并交）
        self.intervals.update(self._build_combined_intervals())

    def _build_singleton_intervals(self, X, y):
        intervals = {}
        for c in self.classes:
            cls_samples = X[y == c]
            intervals[(c,)] = np.column_stack((cls_samples.min(0),
                                               cls_samples.max(0)))  # shape = (3,2)
        return intervals  # dict: key -> ndarray(3,2)

    def _build_combined_intervals(self):
        """
        构造多类别组合假设的区间交集。
        仅保留数值上非空区间（所有维度 low<=high）的组合，否则跳过。
        fixme 这里有可能读出错误的上下限，但目前没有好办法解决，但是最终生成的mass是没有问题的。
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
        针对单个属性生成 BPA，x 是测试样本向量
        """
        sim_dict = {}
        for hyp, mat in self.intervals.items():
            a_interval = [x[attr_idx], x[attr_idx]]  # 退化区间
            b_interval = mat[attr_idx]
            sim_dict[hyp] = similarity(a_interval, b_interval, self.alpha)
        bpa = normalize(sim_dict)
        return bpa


# ========== 调试函数区 ========== #
def show_dataset(X, y, iris):
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['target'] = [iris.target_names[i] for i in y]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print("示例前五行数据:")
    print(df.head())
    print("数据维度:", iris.data.shape)


def show_train_test(X_train, y_train, X_test, y_test, iris):
    iris_feature_names = iris.feature_names
    iris_target_names = iris.target_names
    #
    df_train = pd.DataFrame(X_train, columns=iris_feature_names)
    df_train['target'] = [iris_target_names[i] for i in y_train]
    print("\n训练集（共 %d 个样本）:" % len(df_train))
    print(df_train.head(5))
    print("...")
    print(df_train.tail(5))
    #
    df_test = pd.DataFrame(X_test, columns=iris_feature_names)
    df_test['target'] = [iris_target_names[i] for i in y_test]
    print("\n测试集（共 %d 个样本）:" % len(df_test))
    print(df_test.head(5))
    print("...")
    print(df_test.tail(5))


def print_bpa_sum(bpa, iris):
    """打印 BPA 的各项值与总和，用于验证归一化"""
    total = 0.0
    for k, v in bpa.items():
        total += v
    print(f"  概率和 Σ = {total:.4f}")


def test_interval_generation(model, iris):
    print("\n=== 单类区间数 ===")
    for cls in model.classes:
        key = (cls,)
        print(f"\n{iris.target_names[cls]} ->")
        print(pd.DataFrame(model.intervals[key], columns=['min', 'max'],
                           index=iris.feature_names))
    print("\n=== 组合类交叉区间 ===")
    for hyp in model.intervals:
        if len(hyp) > 1:
            names = [iris.target_names[i] for i in hyp]
            print(f"\n{names} ->")
            print(pd.DataFrame(model.intervals[hyp], columns=['min', 'max'],
                               index=iris.feature_names))


def test_single_bpa(model, sample, iris):
    print("\n=== 单样本各属性的BPA（退化区间） ===")
    for i in range(len(sample)):
        bpa = model.attribute_bpa(sample, i)
        print(f"\n属性 {iris.feature_names[i]}:")
        for k, v in bpa.items():
            labels = ','.join(iris.target_names[i] for i in k)
            print(f"  {labels:<15} : {v:.4f}")
        # 验证mass function和是不是为1
        print_bpa_sum(bpa, iris)


# -------------------------- 主流程复现 -------------------------- #
def experiment(random_state=7, alpha=5.0):
    # 1. 读取 Iris 数据
    iris = load_iris()
    X, y = iris.data, iris.target

    # 2. 随机划分 40/10 规则（论文1设置）
    rs = np.random.RandomState(random_state)
    train_mask = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        train_mask[rs.choice(idx, 40, replace=False)] = True
    test_mask = ~train_mask
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # 测试和调试函数调用
    show_dataset(X, y, iris)
    show_train_test(X_train, y_train, X_test, y_test, iris)

    # 3. 构造区间模型
    model = IntervalBPA(X_train, y_train, alpha=alpha, seed=random_state)

    test_interval_generation(model, iris)
    test_single_bpa(model, X_test[0], iris)

    # 4. 针对每个测试样本生成 4 个属性 BPA 并组合
    y_pred = []
    hypo_singletons = {(c,): iris.target_names[c] for c in model.classes}
    for x in tqdm(X_test, desc="测试样本", ncols=80):
        # 4.1 单属性 BPA 列表
        attr_bpas = [model.attribute_bpa(x, k) for k in range(X.shape[1])]

        # 4.2 Dempster 组合
        fused = attr_bpas[0]
        for bpa in attr_bpas[1:]:
            fused = dempster_combine(fused, bpa)

        # 4.3 预测为最大单类质量
        single_masses = {cls: fused.get((cls,), 0.0) for cls in model.classes}
        pred_cls = max(single_masses, key=single_masses.get)
        y_pred.append(pred_cls)

    acc = accuracy_score(y_test, y_pred)
    return acc, y_test, y_pred


if __name__ == '__main__':
    acc, y_true, y_hat = experiment()
    print(f'\n随机划分一次的准确率: {acc:.3f}')
