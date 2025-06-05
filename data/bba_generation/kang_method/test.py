# data_preparation_drybeans.py
# 复现“区间数 + BPA”生成流程（Dry Beans 数据集）
# 与 data_preparation_iris.py 保持相同接口与注释风格
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------ 参数设置（同步论文 1） ------------------
SEED = 42  # 随机种子
BETA = 5  # 相似度支持系数 β  (式 20)
TOP_K = 5  # “取 5”——只保留 Fisher score 最高的 5 个特征
TRAIN_RATIO = 0.1  # 训练集占比，可按论文 1 调整 0.1~0.9
CLASSES = ['Seker', 'Barbunya', 'Cali', 'Dermason']  # 同论文 1 取 4 类
DATA_PATH = Path('../../DryBeanDataset/Dry_Beans_Dataset.xlsx')  # UCI 原始文件


# ------------------ 工具函数 ------------------
def fisher_score(df: pd.DataFrame, label_col: str):
    """计算每个特征的 Fisher score，返回由大到小排序的特征列表。"""
    scores = {}
    classes = df[label_col].unique()
    for col in df.columns.drop(label_col):
        num, den = 0.0, 0.0
        overall_mean = df[col].mean()
        for c in classes:
            grp = df[df[label_col] == c][col]
            num += len(grp) * (grp.mean() - overall_mean) ** 2
            den += len(grp) * grp.var()
        scores[col] = 0 if den == 0 else num / den
    return sorted(scores, key=scores.get, reverse=True)


def interval_distance(a_low, a_high, b_low, b_high):
    """式 (19) 的区间数欧式距离 D(A,B)。"""
    term1 = ((a_low + a_high) / 2 - (b_low + b_high) / 2) ** 2
    term2 = ((a_high - a_low) + (b_high - b_low)) ** 2 / 12
    return np.sqrt(term1 + term2)


def similarity(dist, beta=BETA):
    """式 (20) 的区间相似度 S(A,B)。"""
    return 1.0 / (1.0 + beta * dist)


def normalize(bpa_vec):
    """式 (21) 归一化，相似度向量 → BPA 向量。"""
    total = bpa_vec.sum()
    return bpa_vec / total if total > 0 else bpa_vec


# ------------------ 数据加载与预处理 ------------------
def load_drybeans():
    df = pd.read_excel(DATA_PATH)
    df = df[df['Class'].isin(CLASSES)].reset_index(drop=True)
    # Fisher score 排序并取前 K 个特征
    best_feats = fisher_score(df, 'Class')[:TOP_K]
    return df, best_feats


def split_train_test(df):
    """按类别 stratify 随机划分训练 / 测试集（比例同步论文 1）。"""
    train_idx, test_idx = [], []
    for c in CLASSES:
        idx = df[df['Class'] == c].index.to_numpy()
        rng = np.random.default_rng(SEED)
        rng.shuffle(idx)
        split = int(len(idx) * TRAIN_RATIO)
        train_idx.extend(idx[:split])
        test_idx.extend(idx[split:])
    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)


# ------------------ BPA 生成主流程 ------------------
def build_interval_model(train_df, feats):
    """Step 1：对每个类-特征求 min/max，构造区间数模型。"""
    model = {}
    for c in CLASSES:
        cls_df = train_df[train_df['Class'] == c]
        for f in feats:
            low, high = cls_df[f].min(), cls_df[f].max()
            model[(c, f)] = (low, high)
    return model


def generate_bpa(sample, model, feats):
    """Step 2~5：把一个测试样本转成 TOP_K 个特征的 BPA 向量字典。"""
    bpa_dict = {}
    for f in feats:
        sim_list = []
        # 对每个类别计算相似度
        for c in CLASSES:
            low, high = model[(c, f)]
            dist = interval_distance(sample[f], sample[f], low, high)
            sim_list.append(similarity(dist))
        bpa_dict[f] = normalize(np.array(sim_list))  # 顺序与 CLASSES 一致
    return bpa_dict


# ------------------ 主函数 ------------------


if __name__ == '__main__':
    # 载入数据并筛选特征
    df, feats = load_drybeans()
    print(f"Top-{TOP_K} features by Fisher score:", feats)
    # 划分训练 / 测试集
    train_df, test_df = split_train_test(df)
    # 构建区间数模型
    model = build_interval_model(train_df, feats)
    # 示例：对测试集中前 3 条样本生成 BPA
    for idx, row in test_df.head(3).iterrows():
        bpa = generate_bpa(row, model, feats)
        print(f"\nSample #{idx} TrueClass={row['Class']}")
        for f in feats:
            print(f"  Feature {f} BPA -> {bpa[f]}")
