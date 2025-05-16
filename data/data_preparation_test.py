import itertools
import math
import random
from collections import defaultdict

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score


# -------------------------- 工具函数 -------------------------- #
def interval_distance(a, b):
    """
    计算区间数 A=[a1,a2], B=[b1,b2] 的距离 D(A,B)
    公式来源：康兵义等(2012) 定义5，简化积分结果
    D^2 = ((a1-b1)^2 + (a1-b1)*(a2-b2) + (a2-b2)^2)/3
    """
    a1, a2 = a
    b1, b2 = b
    d2 = ((a1 - b1) ** 2 + (a1 - b1) * (a2 - b2) + (a2 - b2) ** 2) / 3.0
    return math.sqrt(max(d2, 0.0))


def similarity(a, b, alpha=5.0):
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
                                               cls_samples.max(0)))  # shape = (4,2)
        return intervals  # dict: key -> ndarray(4,2)

    def _build_combined_intervals(self):
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

    # 3. 构造区间模型
    model = IntervalBPA(X_train, y_train, alpha=alpha, seed=random_state)

    # 4. 针对每个测试样本生成 4 个属性 BPA 并组合
    y_pred = []
    hypo_singletons = {(c,): iris.target_names[c] for c in model.classes}
    for x in X_test:
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
    print(f'论文方法复现 — 随机划分一次的准确率: {acc:.3f}')
