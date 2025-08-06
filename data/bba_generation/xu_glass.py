# -*- coding: utf-8 -*-
"""Glass 数据集 BBA 生成器（带函数抽取）
=================================================

严格复现 Xu et al. (2013) 提出的 ``基于正态分布的 BBA 构造`` 算法（去掉证据融合步骤）。

依赖：
    numpy
    pandas
    scipy
    scikit-learn
    torch
    tqdm
输出：
    1. data/xu_bba_glass.csv —— 保存到项目根 ``data`` 目录下（数值已四舍五入到小数点后四位）
    2. 控制台            —— 打印正态性检验表格及部分 BBA 预览（数值四位小数）
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

# ---------- 超参数 ----------
ALPHA = 0.05  # Jarque–Bera 正态性检验显著性水平
TRAIN_RATIO = 0.8  # 可调整的训练集比例
CSV_PATH = Path(__file__).resolve().parents[1] / "xu_bba_glass.csv"
NUM_SAMPLES = 8  # 抽取的样本数
ATTRIBUTES_PER_SAMPLE = 9  # Glass 有 9 个属性
TOTAL_PREVIEW_ROWS = NUM_SAMPLES * ATTRIBUTES_PER_SAMPLE


# ---------- 自定义 Dataset 封装 ----------
class GlassDataset(Dataset):
    """简单封装 Glass 数据集样本与标签"""

    def __init__(self, data: np.ndarray, targets: np.ndarray, attr_names: list, class_names: list):
        self.data = data.astype(np.float32)
        self.targets = targets.astype(np.int64)
        self.attr_names = attr_names
        self.class_names = class_names
        # 记录样本数与属性数
        self.n_samples = data.shape[0]
        self.n_attr = data.shape[1]

    def __len__(self) -> int:  # pragma: no cover - 简单 getter
        return self.n_samples

    def __getitem__(self, idx):  # pragma: no cover - 简单 getter
        # 返回单个样本和对应标签
        return self.data[idx], self.targets[idx]

    def get_attr_names(self):  # pragma: no cover - 简单 getter
        return self.attr_names

    def get_class_names(self):  # pragma: no cover - 简单 getter
        return self.class_names

    def get_sample(self, idx):
        """获取特定索引的原始样本（用于打印）"""
        return self.data[idx], self.targets[idx]


def load_glass_data():
    """读取 Glass 数据集并返回基本信息"""

    data_file = Path(__file__).resolve().parent / "dataset_glass" / "glass.data"
    df_raw = pd.read_csv(data_file, header=None)
    X_all = df_raw.iloc[:, 1:-1].to_numpy(dtype=float)
    y_raw = df_raw.iloc[:, -1].to_numpy(dtype=int)
    label_map = {1: 0, 2: 1, 3: 2, 5: 3, 6: 4, 7: 5}
    y_all = np.array([label_map[int(v)] for v in y_raw], dtype=int)

    attr_names = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
    class_names = ["Bf", "Bn", "Vf", "Co", "Ta", "He"]
    full_class_names = [
        "building_windows_float_processed",
        "building_windows_non_float_processed",
        "vehicle_windows_float_processed",
        "containers",
        "tableware",
        "headlamps",
    ]

    return X_all, y_all, attr_names, class_names, full_class_names


def compute_offsets(X: np.ndarray) -> np.ndarray:
    """计算 Box-Cox 变换所需的平移量"""

    return np.maximum(0, 1e-6 - np.min(X, axis=0))


def normality_test(X: np.ndarray, y: np.ndarray, n_class: int, n_attr: int) -> np.ndarray:
    """返回正态性检验指示矩阵"""

    idx = np.zeros((n_class, n_attr), dtype=int)
    for i in range(n_class):
        cls_mask = y == i
        for j in range(n_attr):
            _, p_val = stats.jarque_bera(X[cls_mask, j])
            idx[i, j] = 1 if p_val < ALPHA else 0
    return idx


def boxcox_single(x, lam):
    """手动实现单值 Box-Cox，``lam == 0`` 时取对数"""

    x = np.maximum(x, 1e-6)
    return np.log(x) if lam == 0 else (x ** lam - 1) / lam


def choose_lambda(sample, attr_idx, mean_vectors, lambdas, transform_flags):
    """Step-1 论文中的预分类（距离法）——仅在该属性需 Box-Cox 时调用"""

    max_vec = np.maximum.reduce([*mean_vectors, sample])
    sample_norm = sample / max_vec
    means_norm = mean_vectors / max_vec
    dists = np.linalg.norm(means_norm - sample_norm, axis=1)
    best_cls = np.argmin(dists)
    return lambdas[best_cls][attr_idx]


def ensure_dir(path: Path):  # pragma: no cover - 简易 IO
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def normalize_round(values: np.ndarray, decimals: int = 4) -> np.ndarray:
    """四舍五入并归一化数组, 保证和为 1。"""

    rounded = np.round(values, decimals)
    diff = round(1.0 - rounded.sum(), decimals)
    if diff != 0:
        idx = int(np.argmax(rounded))
        rounded[idx] = round(rounded[idx] + diff, decimals)
    return rounded


def generate_subset_names(class_names: list) -> list:
    """生成所有可能的命题名称（单元、二元…全集）"""

    subsets = []
    n = len(class_names)
    for r in range(1, n + 1):
        for combo in combinations(class_names, r):
            name = "{" + " ∪ ".join(combo) + "}"
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
        *,
        sample_indices: list[int] | None = None,
        offsets=None,
        decimals: int = 4,
):
    """
    生成 BBA 并返回 ``DataFrame``。列包含 ``sample_index``、``ground_truth``、``dataset_split``、``attribute``、``attribute_data`` 以及所有命题质量。
    所有数值均保留 ``decimals`` 位小数。

    参数 ``sample_indices`` 可提供与 ``train_dataset``、``test_dataset`` 顺序对应的原始数据集索引列表，用于在输出中恢复行号（从 1 开始）。
    参数 ``offsets`` 若提供，则视为在构建数据集时对所有特征施加的平移量，
    ``attribute_data`` 会自动减去对应偏移以恢复原始取值。
    """
    # ---------- Step-3+4: 为每个样本、每个属性生成“嵌套”BBA ----------
    rows = []
    n_attr = len(attr_names)
    n_class = len(class_names)

    total_samples = len(train_dataset) + len(test_dataset)
    if sample_indices is None:
        sample_indices = list(range(total_samples))
    if len(sample_indices) != total_samples:
        raise ValueError("sample_indices 长度必须与数据集样本数一致")

    subset_names = generate_subset_names(class_names)

    for samp_order in tqdm(range(total_samples), desc="Generating BBA", ncols=PROGRESS_NCOLS):
        ds_idx = sample_indices[samp_order]
        if samp_order < len(train_dataset):
            x_vec, y_val = train_dataset.get_sample(samp_order)
        else:
            x_vec, y_val = test_dataset.get_sample(samp_order - len(train_dataset))
        split = "unknown"
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
            masses = normalize_round(w_r / w_r.sum(), decimals)
            mass_dict = {name: 0.0 for name in subset_names}
            for r in range(n_class):
                cls_subset = [class_names[idx] for idx in np.sort(order[: r + 1])]
                name = "{" + " ∪ ".join(cls_subset) + "}"
                mass_dict[name] = float(masses[r])
            # 使用原始数据恢复属性值，避免因平移与四舍五入产生误差
            attr_rec = float(X_all[ds_idx, j])
            row = {
                "sample_index": ds_idx + 1,
                "ground_truth": gt,
                "dataset_split": split,
                "attribute": attr_names[j],
                # 保留更多有效数字，确保与原始数据完全一致
                "attribute_data": round(attr_rec, max(decimals, 5)),
            }
            for name in subset_names:
                row[name] = round(mass_dict[name], decimals)
            rows.append(row)

    return pd.DataFrame(rows)


def preview_sample(df_out: pd.DataFrame, attr_names: list, num_samples: int = NUM_SAMPLES, seed: int = 42) -> None:
    """随机抽取 ``num_samples`` 个样本展示部分 BBA"""

    print(
        f"—— 随机抽取 {num_samples} 个样本对应的 BBA 预览 (每个样本 {len(attr_names)} 行，共 {num_samples * len(attr_names)} 行) ——"
    )
    unique_indices = df_out["sample_index"].unique()
    rng = np.random.RandomState(seed)
    sampled_indices = rng.choice(unique_indices, size=num_samples, replace=False)
    attr_order = {attr: i for i, attr in enumerate(attr_names)}
    for idx in sampled_indices:
        df_group = df_out[df_out["sample_index"] == idx].copy()
        df_group["attr_order"] = df_group["attribute"].map(attr_order)
        df_group = df_group.sort_values(by=["sample_index", "attr_order"])
        df_group = df_group.drop(columns=["attr_order"])
        # 以 Markdown 表格形式输出，便于在支持 Markdown 的环境中查看
        print(df_group.to_markdown(index=False))


def fit_parameters(
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        n_class: int,
        n_attr: int,
) -> tuple[np.ndarray, list[list[float | None]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """根据训练集计算 Box-Cox 及正态模型参数"""

    # ---------- Step-1: 正态性检验 & 需要 Box-Cox 的属性 ----------
    normality_index = normality_test(X_tr, y_tr, n_class, n_attr)
    transform_flags = normality_index.sum(axis=0) >= (n_class / 2)
    print("\n需 Box-Cox 变换的属性:", [attr_names[j] for j, f in enumerate(transform_flags) if f])

    lambdas: list[list[float | None]] = [[None] * n_attr for _ in range(n_class)]
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

    # ---------- Step-2: 建立“正态分布模型” μ_ij, σ_ij ----------
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
    return transform_flags, lambdas, mus, sigmas, mean_vectors, normality_index


def generate_and_save_bba(
        X_all: np.ndarray,
        y_all: np.ndarray,
        attr_names: list,
        class_names: list,
        full_class_names: list,
        csv_path: Path = CSV_PATH,
        train_ratio: float = TRAIN_RATIO,
        decimals: int = 4,
) -> pd.DataFrame:
    """生成 BBA ``DataFrame`` 并保存至 ``csv_path``"""

    n_class = len(class_names)
    n_attr = len(attr_names)

    indices = np.arange(len(X_all))
    train_idx, test_idx = train_test_split(
        indices, train_size=train_ratio, stratify=y_all, random_state=0
    )
    X_tr = X_all[train_idx].copy()
    y_tr = y_all[train_idx].copy()
    X_te = X_all[test_idx].copy()
    y_te = y_all[test_idx].copy()

    train_dataset = GlassDataset(X_tr, y_tr, attr_names, class_names)
    test_dataset = GlassDataset(X_te, y_te, attr_names, class_names)

    offsets = compute_offsets(X_tr)
    X_tr += offsets
    X_te += offsets

    (
        transform_flags,
        lambdas,
        mus,
        sigmas,
        mean_vectors,
        normality_index,
    ) = fit_parameters(X_tr, y_tr, n_class, n_attr)

    df_norm = pd.DataFrame(
        normality_index,
        index=full_class_names,
        columns=attr_names,
    )

    print("\nTraining Set Normality Indices (1 表示拒绝正态假设):")
    print(df_norm.to_string())
    print("\n需 Box-Cox 变换的属性:", [attr_names[j] for j, f in enumerate(transform_flags) if f])

    dataset_order = list(train_idx) + list(test_idx)
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
        sample_indices=dataset_order,
        offsets=offsets,
        decimals=decimals,
    )

    attr_order = {attr: i for i, attr in enumerate(attr_names)}
    df_out["_attr_order"] = df_out["attribute"].map(attr_order)
    df_out = df_out.sort_values(by=["sample_index", "_attr_order"]).reset_index(drop=True)
    df_out = df_out.drop(columns="_attr_order")

    ensure_dir(csv_path)
    df_out.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n已保存格式化后的 BBA 至 {csv_path.resolve()}  (共 {len(df_out)} 行)\n")
    return df_out


if __name__ == "__main__":
    X_all, y_all, attr_names, class_names, full_class_names = load_glass_data()
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

    df_out = generate_and_save_bba(
        X_all,
        y_all,
        attr_names,
        class_names,
        full_class_names,
        csv_path=CSV_PATH,
        train_ratio=TRAIN_RATIO,
        decimals=4,
    )

    preview_sample(df_out, attr_names)
