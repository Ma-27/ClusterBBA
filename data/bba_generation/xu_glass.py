# -*- coding: utf-8 -*-
"""
Glass 数据集 BBA 生成器
严格复现 Xu et al. (2013) 提出的基于正态分布的 BBA 构造算法（去掉证据融合步骤）。
--------------------------------------------------------------------
依赖：
    numpy
    pandas
    scipy
    scikit-learn
    torch
    tqdm
输出：
    1. data/xu_bba_glass.csv —— 保存到项目根 data 目录下（数值已四舍五入到小数点后四位）
    2. 控制台            —— 打印正态性检验表格及前若干条 BBA（数值四位小数）
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

# 确保包导入路径指向项目根目录
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# 依赖本项目内现成工具函数 / 模块
from config import PROGRESS_NCOLS


class GlassDataset(Dataset):
    """简单封装 Glass 数据集样本与标签"""

    def __init__(self, data: np.ndarray, targets: np.ndarray, attr_names: list, class_names: list):
        self.data = data.astype(np.float32)
        self.targets = targets.astype(np.int64)
        self.attr_names = attr_names
        self.class_names = class_names
        self.n_samples = data.shape[0]
        self.n_attr = data.shape[1]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

    def get_attr_names(self):
        return self.attr_names

    def get_class_names(self):
        return self.class_names

    def get_sample(self, idx):
        return self.data[idx], self.targets[idx]


# ---------- 超参数 ----------
ALPHA = 0.05  # Jarque–Bera 正态性检验显著性水平
TRAIN_RATIO = 0.8
CSV_PATH = Path(__file__).resolve().parents[1] / "xu_bba_glass.csv"
NUM_SAMPLES = 8


def boxcox_single(x, lam):
    """手动实现单值 Box-Cox，lam == 0 时取对数"""
    x = np.maximum(x, 1e-6)
    return np.log(x) if lam == 0 else (x ** lam - 1) / lam


def choose_lambda(sample, attr_idx, mean_vectors, lambdas, transform_flags):
    """Step-1 论文中的预分类(距离法)——仅在该属性需 Box-Cox 时才调用"""
    max_vec = np.maximum.reduce([*mean_vectors, sample])
    sample_norm = sample / max_vec
    means_norm = mean_vectors / max_vec
    dists = np.linalg.norm(means_norm - sample_norm, axis=1)
    best_cls = np.argmin(dists)
    return lambdas[best_cls][attr_idx]


def ensure_dir(path: Path):
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def generate_subset_names(class_names: list) -> list:
    """生成所有可能的命题名称（单元、二元…全集）"""
    subsets = []
    n = len(class_names)
    for r in range(1, n + 1):
        for combo in combinations(class_names, r):
            name = "{" + " ∪ ".join(sorted(combo)) + "}"
            subsets.append(name)
    return subsets


def generate_bba_dataframe(
        X_all,
        y_all,
        X_tr,
        y_tr,
        train_dataset,
        test_dataset,
        attr_names,
        class_names,
        full_class_names,
        transform_flags,
        lambdas,
        mus,
        sigmas,
        mean_vectors,
):
    """
    生成 BBA 并返回 DataFrame。
    列包含 ['sample_index','ground_truth','dataset_split','attribute','attribute_data', ... 所有命题 ...]
    所有数值均保留四位小数。
    """
    rows = []
    n_attr = len(attr_names)
    total_samples = len(train_dataset) + len(test_dataset)
    subset_names = generate_subset_names(class_names)

    for samp_idx in tqdm(range(total_samples), desc="Generating BBA", ncols=PROGRESS_NCOLS):
        if samp_idx < len(train_dataset):
            x_vec, y_val = train_dataset.get_sample(samp_idx)
        else:
            x_vec, y_val = test_dataset.get_sample(samp_idx - len(train_dataset))
        split = 'unknown'
        gt = full_class_names[y_val]

        for j in range(n_attr):
            x_val = x_vec[j]
            if transform_flags[j]:
                lam = choose_lambda(x_vec, j, mean_vectors, lambdas, transform_flags)
                x_val_trans = boxcox_single(x_val, lam)
            else:
                x_val_trans = x_val
            pdf_vals = stats.norm.pdf(x_val_trans, loc=mus[:, j], scale=sigmas[:, j])
            order = np.argsort(pdf_vals)[::-1]
            w_r = pdf_vals[order]
            masses = w_r / w_r.sum()
            mass_dict = {name: 0.0 for name in subset_names}
            for r in range(len(class_names)):
                cls_subset = [class_names[idx] for idx in np.sort(order[: r + 1])]
                name = "{" + " ∪ ".join(cls_subset) + "}"
                mass_dict[name] = float(masses[r])
            row = {
                'sample_index': samp_idx,
                'ground_truth': gt,
                'dataset_split': split,
                'attribute': attr_names[j],
                'attribute_data': round(float(X_all[samp_idx, j]), 4),
            }
            for name in subset_names:
                row[name] = round(mass_dict[name], 4)
            rows.append(row)

    return pd.DataFrame(rows)


def preview_sample(df_out: pd.DataFrame, attr_names: list, num_samples: int = 8, seed: int = 42) -> None:
    """随机抽取样本展示部分 BBA"""
    print(
        f"—— 随机抽取 {num_samples} 个样本对应的 BBA 预览 (每个样本 {len(attr_names)} 行，共 {num_samples * len(attr_names)} 行) ——"
    )
    unique_indices = df_out['sample_index'].unique()
    rng = np.random.RandomState(seed)
    sampled_indices = rng.choice(unique_indices, size=num_samples, replace=False)
    attr_order = {attr: i for i, attr in enumerate(attr_names)}
    for idx in sampled_indices:
        df_group = df_out[df_out['sample_index'] == idx].copy()
        df_group['attr_order'] = df_group['attribute'].map(attr_order)
        df_group = df_group.sort_values(by=['sample_index', 'attr_order'])
        df_group = df_group.drop(columns=['attr_order'])
        print(df_group.to_string(index=False))


if __name__ == '__main__':
    # ---------- Step-0: 读取并准备基础数据 ----------
    data_file = Path(__file__).resolve().parent / 'dataset_glass' / 'glass.data'
    raw = pd.read_csv(data_file, header=None)
    X_all = raw.iloc[:, 1:-1].values
    y_raw = raw.iloc[:, -1].values
    label_map = {1: 0, 2: 1, 3: 2, 5: 3, 6: 4, 7: 5}
    y_all = np.array([label_map[int(v)] for v in y_raw])
    attr_names = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
    class_names = ['Bf', 'Bn', 'Vf', 'Co', 'Ta', 'He']
    full_class_names = [
        'building_windows_float_processed',
        'building_windows_non_float_processed',
        'vehicle_windows_float_processed',
        'containers',
        'tableware',
        'headlamps',
    ]
    n_class = len(class_names)
    n_attr = X_all.shape[1]

    print(f"数据集包含 {n_class} 类，{n_attr} 个属性。\n")
    for i in range(5):
        sample_vals = np.round(X_all[i], 4)
        label = full_class_names[y_all[i]]
        print(f"样本 {i} - 特征: {sample_vals.tolist()}，标签: {label}")
    print(f"总样本数（全体 X）= {X_all.shape[0]}")
    print(f"类别标签（full_class_names）: {full_class_names}")
    print(f"属性名称（attr_names）: {attr_names}\n")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, train_size=TRAIN_RATIO, stratify=y_all, random_state=0
    )

    train_dataset = GlassDataset(X_tr, y_tr, attr_names, class_names)
    test_dataset = GlassDataset(X_te, y_te, attr_names, class_names)

    offsets = np.maximum(0, 1e-6 - np.min(X_tr, axis=0))
    X_tr += offsets
    X_te += offsets

    normality_index = np.zeros((n_class, n_attr), dtype=int)

    for i in range(n_class):
        cls_mask = y_tr == i
        for j in range(n_attr):
            jb_stat, p_val = stats.jarque_bera(X_tr[cls_mask, j])
            normality_index[i, j] = 1 if p_val < ALPHA else 0

    df_norm = pd.DataFrame(
        normality_index,
        index=full_class_names,
        columns=attr_names,
    )
    print("\nTraining Set Normality Indices (1 表示拒绝正态假设):")
    print(df_norm.to_string())

    transform_flags = normality_index.sum(axis=0) >= (n_class / 2)
    print("\n需 Box-Cox 变换的属性:", [attr_names[j] for j, f in enumerate(transform_flags) if f])

    lambdas = [[None] * n_attr for _ in range(n_class)]
    for j, need_tf in enumerate(transform_flags):
        if not need_tf:
            continue
        for i in range(n_class):
            cls_data = X_tr[y_tr == i, j]
            if np.allclose(cls_data, cls_data[0]):
                # 该类该属性全为常数，无法估计 Box-Cox λ
                lambdas[i][j] = 1.0
            else:
                _, lam = stats.boxcox(cls_data, lmbda=None)
                lambdas[i][j] = lam

    mus = np.zeros((n_class, n_attr))
    sigmas = np.zeros((n_class, n_attr))
    for i in tqdm(range(n_class), desc="Calculating means and stds", ncols=PROGRESS_NCOLS):
        cls_mask = y_tr == i
        for j in range(n_attr):
            data = X_tr[cls_mask, j]
            if transform_flags[j]:
                lam = lambdas[i][j]
                data = boxcox_single(data, lam)
            mus[i, j] = data.mean()
            sigma = data.std(ddof=1)
            sigmas[i, j] = sigma if sigma > 0 else 1e-6

    mean_vectors = np.vstack([X_tr[y_tr == i].mean(axis=0) for i in range(n_class)])

    df_out = generate_bba_dataframe(
        X_all,
        y_all,
        X_tr,
        y_tr,
        train_dataset,
        test_dataset,
        attr_names,
        class_names,
        full_class_names,
        transform_flags,
        lambdas,
        mus,
        sigmas,
        mean_vectors,
    )

    ensure_dir(CSV_PATH)
    df_out.to_csv(CSV_PATH, index=False, encoding='utf-8')
    print(f"\n已保存格式化后的 BBA 至 {CSV_PATH.resolve()}  (共 {len(df_out)} 行)\n")

    preview_sample(df_out, attr_names)
