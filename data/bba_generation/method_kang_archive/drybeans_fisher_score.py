import math
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ------------------ 参数设置（同步论文 1） ------------------
SEED = 42  # 随机种子
BETA = 5  # 相似度支持系数 β  (式 20)
TOP_K = 5  # “取 5”——只保留 Fisher score 最高的 5 个特征
TRAIN_RATIO = 0.1  # 训练集占比，可按论文 1 调整 0.1~0.9
CLASSES = ['SEKER', 'BARBUNYA', 'CALI', 'DERMASON']  # 与论文 1 保持一致，仅取 4 类
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / 'dataset_dry_bean'
DATA_PATH = DATA_DIR / 'Dry_Beans_Dataset.xlsx'  # UCI 原始文件
CSV_PATH = DATA_DIR / 'Dry_Beans_Dataset.csv'  # 本地缓存的 CSV 文件


# ------------------ Fisher score 相关 ------------------

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


# ------------------ 区间数距离 / 相似度 ------------------

def interval_distance(a_low, a_high, b_low, b_high):
    """式 (19) 的区间数欧氏距离 D(A,B)。"""
    term1 = ((a_low + a_high) / 2 - (b_low + b_high) / 2) ** 2
    term2 = ((a_high - a_low) + (b_high - b_low)) ** 2 / 12
    return math.sqrt(term1 + term2)


def similarity(dist, beta=BETA):
    """式 (20) 的区间相似度 S(A,B)。"""
    return 1.0 / (1.0 + beta * dist)


# ------------------ 辅助工具 ------------------

def normalize(vec: np.ndarray):
    """式 (21) 归一化：相似度向量 → BPA 向量。"""
    s = vec.sum()
    return vec / s if s > 0 else vec


# ------------------ 数据集下载 & 预处理 ------------------

def download_dry_beans():
    """若本地无数据，则从 UCI 下载并转换为 CSV。"""
    if CSV_PATH.exists():
        return CSV_PATH

    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00602/DryBeanDataset.zip'
    print('开始下载 Dry Beans 数据集...')
    resp = requests.get(url)
    resp.raise_for_status()
    with zipfile.ZipFile(BytesIO(resp.content)) as zf:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        zf.extractall(DATA_DIR)
        xlsx = next(f for f in zf.namelist() if f.endswith('.xlsx'))
        df = pd.read_excel(DATA_DIR / xlsx)
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CSV_PATH, index=False)
        print('已保存 CSV ->', CSV_PATH)
    return CSV_PATH


def load_drybeans():
    """加载 Dry Beans 数据集并筛选前 TOP_K 特征。"""
    csv_path = download_dry_beans()
    df = pd.read_csv(csv_path)
    df = df[df['Class'].isin(CLASSES)].reset_index(drop=True)
    best_feats = fisher_score(df, 'Class')[:TOP_K]
    return df, best_feats


def split_train_test(df: pd.DataFrame):
    """按类别 stratify 划分训练 / 测试集。"""
    train_idx, test_idx = [], []
    rng = np.random.default_rng(SEED)
    for c in CLASSES:
        idx = df[df['Class'] == c].index.to_numpy()
        rng.shuffle(idx)
        split = int(len(idx) * TRAIN_RATIO)
        train_idx.extend(idx[:split])
        test_idx.extend(idx[split:])
    return df.loc[train_idx].reset_index(drop=True), df.loc[test_idx].reset_index(drop=True)


# ------------------ 区间模型 & BPA 生成 ------------------

def build_interval_model(train_df: pd.DataFrame, feats):
    """Step‑1：对每个 (类, 特征) 计算 min/max，形成区间数模型。"""
    model = {}
    for c in CLASSES:
        cls_df = train_df[train_df['Class'] == c]
        for f in feats:
            low, high = cls_df[f].min(), cls_df[f].max()
            model[(c, f)] = (low, high)
    return model


def generate_bpa(sample: pd.Series, model: dict, feats):
    """Step‑2~5：将单个测试样本转换为 TOP_K 特征的 BPA 向量字典。"""
    bpa_dict = {}
    for f in feats:
        sim_vec = []
        for c in CLASSES:
            low, high = model[(c, f)]
            dist = interval_distance(sample[f], sample[f], low, high)
            sim_vec.append(similarity(dist))
        bpa_dict[f] = normalize(np.array(sim_vec))  # 顺序与 CLASSES 一致
    return bpa_dict


# ------------------ 主函数 ------------------

if __name__ == '__main__':
    # 1. 载入数据 + 特征筛选
    df, feats = load_drybeans()
    print(f"Top-{TOP_K} features by Fisher score: {feats}\n")

    # 2. 划分训练 / 测试集
    train_df, test_df = split_train_test(df)
    print(f"Train size = {len(train_df)}, Test size = {len(test_df)}\n")

    # 3. 构建区间数模型
    model = build_interval_model(train_df, feats)

    # 4. 对测试集所有样本生成 BPA
    bpa_results = {}
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Generating BPA'):
        bpa_results[idx] = generate_bpa(row, model, feats)

    # 5. 打印前 5 条样本的 BPA 结果示例
    for i, (idx, bpa) in enumerate(bpa_results.items()):
        if i >= 5:
            break
        print(f"\nSample #{idx} TrueClass={test_df.loc[idx, 'Class']}")
        for f in feats:
            print(f"  Feature {f} BPA -> {bpa[f]}")

    print(f"\n已为 {len(bpa_results)} 个测试样本生成 BPA ✨")
