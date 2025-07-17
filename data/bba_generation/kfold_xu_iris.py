# -*- coding: utf-8 -*-
"""
Iris 数据集 BBA 生成器（带函数抽取）
严格复现 Xu et al. (2013) 提出的“基于正态分布的 BBA 构造”算法（去掉后续证据融合步骤）。
--------------------------------------------------------------------
依赖：
    numpy
    pandas
    scipy
    scikit-learn
    torch
    tqdm
输出：
    data/iris_fold_<fold>_bba.csv —— 保存到项目根 data 目录下（5 折，每折生成一个文件，数值已四舍五入到小数点后四位）
    控制台            —— 打印每折正态性检验表格及生成提示
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from tqdm import tqdm


# ---------- 自定义 Dataset 封装 ----------
class IrisDataset(Dataset):
    """简单封装 Iris 数据集样本与标签"""

    def __init__(self, data: np.ndarray, targets: np.ndarray, attr_names: list, class_names: list):
        self.data = data.astype(np.float32)
        self.targets = targets.astype(np.int64)
        self.attr_names = attr_names
        self.class_names = class_names
        # 记录样本数与属性数
        self.n_samples = data.shape[0]
        self.n_attr = data.shape[1]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 返回单个样本和对应标签
        return self.data[idx], self.targets[idx]

    def get_attr_names(self):
        return self.attr_names

    def get_class_names(self):
        return self.class_names

    def get_sample(self, idx):
        # 获取特定索引的原始样本（用来打印）
        return self.data[idx], self.targets[idx]


# ---------- 超参数 ----------
ALPHA = 0.05  # Jarque–Bera 正态性检验显著性水平
N_FOLDS = 5  # 5 折交叉验证
# 保存到项目根目录下的 data 文件夹
CSV_DIR = Path(__file__).resolve().parents[1] / "data"
CSV_DIR.mkdir(parents=True, exist_ok=True)

PRINT_ROWS = 24  # 打印多少行到控制台


def boxcox_single(x, lam):
    """手动实现单值 Box-Cox，lam == 0 时取对数"""
    return np.log(x) if lam == 0 else (x ** lam - 1) / lam


def choose_lambda(sample, attr_idx, mean_vectors, lambdas, transform_flags):
    """Step-1 论文中的预分类(距离法)——仅在该属性需 Box-Cox 时才调用"""
    # 距离四 (归一化欧氏距离)
    max_vec = np.maximum.reduce([*mean_vectors, sample])
    sample_norm = sample / max_vec
    means_norm = mean_vectors / max_vec
    dists = np.linalg.norm(means_norm - sample_norm, axis=1)
    best_cls = np.argmin(dists)
    return lambdas[best_cls][attr_idx]


def ensure_dir(path: Path):
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def generate_bba_dataframe(X_all, y_all, train_idx, test_idx,
                           attr_names, class_names, full_class_names,
                           transform_flags, lambdas, mus, sigmas, mean_vectors,
                           fold_idx):
    """
    生成 BBA 并返回 DataFrame，包含列：
    ['fold','sample_index','ground_truth','dataset_split','attribute','attribute_data', '{Vi}', '{Ve}', '{Se}',
     '{Vi ∪ Ve}', '{Vi ∪ Se}', '{Ve ∪ Se}', '{Vi ∪ Ve ∪ Se}']
    所有数值均保留四位小数。
    """
    n_attr = len(attr_names)
    train_dataset = IrisDataset(X_all[train_idx], y_all[train_idx], attr_names, class_names)
    test_dataset = IrisDataset(X_all[test_idx], y_all[test_idx], attr_names, class_names)
    rows = []
    total_samples = len(train_dataset) + len(test_dataset)

    # 使用 Dataset 迭代样本
    for samp_idx in tqdm(range(total_samples), desc=f"Fold {fold_idx} BBA"):
        if samp_idx < len(train_dataset):
            x_vec, y_val = train_dataset.get_sample(samp_idx)
            split = 'train'
            global_idx = train_idx[samp_idx]
        else:
            x_vec, y_val = test_dataset.get_sample(samp_idx - len(train_dataset))
            split = 'test'
            global_idx = test_idx[samp_idx - len(train_dataset)]
        gt = full_class_names[y_val]

        # 对每个属性生成 BBA
        for j in range(n_attr):
            x_val = x_vec[j]
            # 对属性值进行 Box-Cox 转换（若需要）
            if transform_flags[j]:
                lam = choose_lambda(x_vec, j, mean_vectors, lambdas, transform_flags)
                x_val_trans = boxcox_single(x_val, lam)
            else:
                x_val_trans = x_val
            # 计算 n 类 正态密度值
            pdf_vals = stats.norm.pdf(x_val_trans, loc=mus[:, j], scale=sigmas[:, j])
            # 按降序排序并构造嵌套子集
            order = np.argsort(pdf_vals)[::-1]
            w_r = pdf_vals[order]
            masses = w_r / w_r.sum()
            # 定义七种命题名称对应顺序：
            # {Vi}, {Ve}, {Se}, {Vi ∪ Ve}, {Vi ∪ Se}, {Ve ∪ Se}, {Vi ∪ Ve ∪ Se}
            mass_dict = {'{Vi}': 0.0, '{Ve}': 0.0, '{Se}': 0.0,
                         '{Vi ∪ Ve}': 0.0, '{Vi ∪ Se}': 0.0, '{Ve ∪ Se}': 0.0,
                         '{Vi ∪ Ve ∪ Se}': 0.0}
            # r=0: 单元集 {class_names[order[0]]}
            cls0 = class_names[order[0]]
            mass_dict[f'{{{cls0}}}'] = float(masses[0])
            # r=1: 二元集 {order[0], order[1]}
            cls_a, cls_b = class_names[order[0]], class_names[order[1]]
            pair_set = set([cls_a, cls_b])
            if pair_set == set(['Vi', 'Ve']):
                mass_dict['{Vi ∪ Ve}'] = float(masses[1])
            elif pair_set == set(['Vi', 'Se']):
                mass_dict['{Vi ∪ Se}'] = float(masses[1])
            elif pair_set == set(['Ve', 'Se']):
                mass_dict['{Ve ∪ Se}'] = float(masses[1])
            # r=2: 三元集 {Vi, Ve, Se}
            mass_dict['{Vi ∪ Ve ∪ Se}'] = float(masses[2])

            # 构造行并保留四位小数
            row = {
                'fold': fold_idx,
                'sample_index': global_idx,
                'ground_truth': gt,
                'dataset_split': split,
                'attribute': attr_names[j],
                'attribute_data': round(float(x_vec[j]), 4),
                '{Vi}': round(mass_dict['{Vi}'], 4),
                '{Ve}': round(mass_dict['{Ve}'], 4),
                '{Se}': round(mass_dict['{Se}'], 4),
                '{Vi ∪ Ve}': round(mass_dict['{Vi ∪ Ve}'], 4),
                '{Vi ∪ Se}': round(mass_dict['{Vi ∪ Se}'], 4),
                '{Ve ∪ Se}': round(mass_dict['{Ve ∪ Se}'], 4),
                '{Vi ∪ Ve ∪ Se}': round(mass_dict['{Vi ∪ Ve ∪ Se}'], 4)
            }
            rows.append(row)

    df_bba = pd.DataFrame(rows)
    df_bba.to_csv(CSV_DIR / f"iris_fold_{fold_idx}_bba.csv", index=False)
    print(f"Fold {fold_idx}: saved {len(df_bba)} BBA rows to CSV.")
    return df_bba


if __name__ == '__main__':
    # ---------- Step-0: 读取并准备基础数据 ----------
    iris = load_iris()
    X_all = iris["data"].copy()
    y_all = iris["target"].copy()
    attr_names = ["SL", "SW", "PL", "PW"]  # 分别对应 sepal length, sepal width, petal length, petal width
    class_names = ["Se", "Ve", "Vi"]
    full_class_names = ["Setosa", "Versicolor", "Virginica"]  # 对应完整名称
    n_class = len(class_names)
    n_attr = X_all.shape[1]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
    fold_idx = 0
    for train_idx, test_idx in skf.split(X_all, y_all):
        print(f"\n=== Fold {fold_idx} ===")
        X_tr, X_te = X_all[train_idx].copy(), X_all[test_idx].copy()
        y_tr, y_te = y_all[train_idx].copy(), y_all[test_idx].copy()

        # 为了后续 Box-Cox，保证所有数值严格为正：若 min ≤0，则整体平移
        offsets = np.maximum(0, 1e-6 - np.min(X_tr, axis=0))
        X_tr += offsets
        X_te += offsets

        # ---------- Step-1: 正态性检验 & 需要 Box-Cox 的属性 ----------
        normality_index = np.zeros((n_class, n_attr), dtype=int)

        for i in range(n_class):
            cls_mask = y_tr == i
            for j in range(n_attr):
                jb_stat, p_val = stats.jarque_bera(X_tr[cls_mask, j])
                normality_index[i, j] = 1 if p_val < ALPHA else 0

        # 构造并打印正态性检验结果表格
        df_norm = pd.DataFrame(
            normality_index,
            index=full_class_names,
            columns=attr_names
        )
        print("\nTraining Set Normality Indices (1 表示拒绝正态假设):")
        print(df_norm.to_string())

        # 条件 3：若“非正态”类 ≥ 一半，则对该属性整体做 Box-Cox
        transform_flags = normality_index.sum(axis=0) >= (n_class / 2)
        print("需 Box–Cox 变换的属性:", [attr_names[j] for j, f in enumerate(transform_flags) if f])

        # 计算每个“需要变换的属性-类别”对应的 λ（论文记作 k_{ij}）
        lambdas = [[None] * n_attr for _ in range(n_class)]  # 若不需变换则 None
        for j, need_tf in enumerate(transform_flags):
            if not need_tf:
                continue
            for i in range(n_class):
                cls_data = X_tr[y_tr == i, j]
                _, lam = stats.boxcox(cls_data, lmbda=None)
                lambdas[i][j] = lam

        # ---------- Step-2: 建立“正态分布模型” μ_ij, σ_ij ----------
        mus = np.zeros((n_class, n_attr))
        sigmas = np.zeros((n_class, n_attr))
        for i in tqdm(range(n_class), desc="Calculating means and stds"):
            cls_mask = y_tr == i
            for j in range(n_attr):
                data = X_tr[cls_mask, j]
                if transform_flags[j]:
                    lam = lambdas[i][j]
                    data = boxcox_single(data, lam)
                mus[i, j] = data.mean()
                sigmas[i, j] = data.std(ddof=1)

        # 预先计算各类别在“原始特征空间”中的均值向量，用于选 λ
        mean_vectors = np.vstack([X_tr[y_tr == i].mean(axis=0) for i in range(n_class)])

        # 生成 BBA 并写入 CSV
        generate_bba_dataframe(
            X_all, y_all, train_idx, test_idx,
            attr_names, class_names, full_class_names,
            transform_flags, lambdas, mus, sigmas, mean_vectors,
            fold_idx
        )
        fold_idx += 1

    print("\n5 折交叉验证 BBA 生成完成。")
