# -*- coding: utf-8 -*-
"""
Iris 数据集 BBA 生成器（带函数抽取），严格复现 Xu et al. (2013) 提出的“基于正态分布的 BBA 构造”算法（无后续证据融合步骤）。
--------------------------------------------------------------------
依赖：
    numpy
    pandas
    scipy
    scikit-learn
    torch
    tqdm
输出：
    1. data/xu_bba_iris.csv —— 保存到项目根 data 目录下（数值已四舍五入到小数点后四位）
    2. 控制台            —— 打印正态性检验表格及前若干条 BBA（数值四位小数）
"""

import sys
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
TRAIN_RATIO = 0.8  # 40/10 的随机划分(DBE)，可自行调整
# 保存到项目根目录下的 data 文件夹
CSV_PATH = Path(__file__).resolve().parents[1] / "xu_bba_iris.csv"
NUM_SAMPLES = 8  # 抽取的样本数
ATTRIBUTES_PER_SAMPLE = 4  # 每个样本的属性行数
TOTAL_PREVIEW_ROWS = NUM_SAMPLES * ATTRIBUTES_PER_SAMPLE


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


def load_iris_data():
    """读取 Iris 数据集并返回基本信息"""

    data_file = Path(__file__).resolve().parent / "dataset_iris" / "iris.data"
    df_raw = pd.read_csv(
        data_file,
        header=None,
        names=["SL", "SW", "PL", "PW", "label"],
    )
    label_map = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2,
    }
    X_all = df_raw[["SL", "SW", "PL", "PW"]].to_numpy(dtype=float)
    y_all = df_raw["label"].map(label_map).to_numpy(dtype=int)
    attr_names = ["SL", "SW", "PL", "PW"]
    class_names = ["Se", "Ve", "Vi"]
    full_class_names = ["Setosa", "Versicolor", "Virginica"]
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


def normalize_round(values: np.ndarray, decimals: int = 4) -> np.ndarray:
    """四舍五入并归一化数组, 保证和为 1。"""

    rounded = np.round(values, decimals)
    diff = round(1.0 - rounded.sum(), decimals)
    if diff != 0:
        idx = int(np.argmax(rounded))
        rounded[idx] = round(rounded[idx] + diff, decimals)
    return rounded


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
    生成 BBA 并返回 DataFrame，包含列：
    ['sample_index','ground_truth','dataset_split','attribute','attribute_data', '{Vi}', '{Ve}', '{Se}',
     '{Vi ∪ Ve}', '{Vi ∪ Se}', '{Ve ∪ Se}', '{Vi ∪ Ve ∪ Se}']
    所有数值均保留四位小数。

    参数 ``sample_indices`` 可提供与 ``train_dataset``、``test_dataset`` 顺序对应的原始数据集索引列表，用于在输出中恢复行号（从 1 开始）。
    参数 ``offsets`` 若提供，则视为在构建数据集时对所有特征施加的平移量，
    ``attribute_data`` 会自动减去对应偏移以恢复原始取值。
    """
    # ---------- Step-3+4: 为每个样本、每个属性生成“嵌套”BBA ----------
    # 这里记录每个 sample_index, dataset_split, ground_truth, attribute, attribute_data, 以及 7个质量列
    rows = []
    n_attr = len(attr_names)

    total_samples = len(train_dataset) + len(test_dataset)
    if sample_indices is None:
        sample_indices = list(range(total_samples))
    if len(sample_indices) != total_samples:
        raise ValueError("sample_indices 长度必须与数据集样本数一致")

    # 逐样本生成各属性的 BBA，进度条展示整体进度
    for samp_order in tqdm(
            range(total_samples), desc="Generating BBA", ncols=PROGRESS_NCOLS):
        ds_idx = sample_indices[samp_order]
        if samp_order < len(train_dataset):
            x_vec, y_val = train_dataset.get_sample(samp_order)
            # split = 'train'
        else:
            x_vec, y_val = test_dataset.get_sample(samp_order - len(train_dataset))
            # split = 'test'
        # 统一标记数据集划分为 unknown
        split = 'unknown'
        gt = full_class_names[y_val]

        # 注意：这里并不需要知道真实 label，就算有也不做融合
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
            # 强制归一化质量
            masses = normalize_round(w_r / w_r.sum(), decimals)
            # 定义七种命题名称对应顺序：
            # {Vi}, {Ve}, {Se}, {Vi ∪ Ve}, {Vi ∪ Se}, {Ve ∪ Se}, {Vi ∪ Ve ∪ Se}
            # 初始化质量字典，键与输出列名保持一致
            mass_dict = {
                '{Vi}': 0.0, '{Ve}': 0.0, '{Se}': 0.0,
                '{Vi ∪ Ve}': 0.0, '{Vi ∪ Se}': 0.0, '{Ve ∪ Se}': 0.0, '{Vi ∪ Ve ∪ Se}': 0.0
            }
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
            # 对于第 r>=2 仅填前三，因为 BBA 只生成 3 层嵌套。若想保留全部 7，可修改此处逻辑。

            # 构造行并保留四位小数
            attr_rec = x_vec[j]
            if offsets is not None:
                attr_rec -= offsets[j]
            row = {
                'sample_index': ds_idx + 1,
                'ground_truth': gt,
                'dataset_split': split,
                'attribute': attr_names[j],
                'attribute_data': round(float(attr_rec), decimals),
                '{Vi}': round(mass_dict['{Vi}'], decimals),
                '{Ve}': round(mass_dict['{Ve}'], decimals),
                '{Se}': round(mass_dict['{Se}'], decimals),
                '{Vi ∪ Ve}': round(mass_dict['{Vi ∪ Ve}'], decimals),
                '{Vi ∪ Se}': round(mass_dict['{Vi ∪ Se}'], decimals),
                '{Ve ∪ Se}': round(mass_dict['{Ve ∪ Se}'], decimals),
                '{Vi ∪ Ve ∪ Se}': round(mass_dict['{Vi ∪ Ve ∪ Se}'], decimals)
            }
            rows.append(row)

    return pd.DataFrame(rows)


def preview_sample(df_out: pd.DataFrame, attr_names: list, num_samples: int = 8, seed: int = 42) -> None:
    """
    随机抽取 `num_samples` 个 `sample_index`，并按属性顺序展示对应的 BBA。
    默认每个样本展示 4 条，属性顺序按 `attr_names`。
    """
    print(
        f"—— 随机抽取 {num_samples} 个样本对应的 BBA 预览 (每个样本 {len(attr_names)} 行，共 {num_samples * len(attr_names)} 行) ——")
    unique_indices = df_out['sample_index'].unique()
    rng = np.random.RandomState(seed)
    sampled_indices = rng.choice(unique_indices, size=num_samples, replace=False)
    # 按顺序循环打印每个样本的 BBA，组之间空行分隔
    attr_order = {attr: i for i, attr in enumerate(attr_names)}
    for idx in sampled_indices:
        df_group = df_out[df_out['sample_index'] == idx].copy()
        df_group['attr_order'] = df_group['attribute'].map(attr_order)
        df_group = df_group.sort_values(by=['sample_index', 'attr_order'])
        df_group = df_group.drop(columns=['attr_order'])
        print(df_group.to_string(index=False))


def fit_parameters(
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        n_class: int,
        n_attr: int,
) -> tuple[np.ndarray, list[list[float | None]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """根据训练集计算 Box-Cox 及正态模型参数"""

    # ---------- Step-1: 正态性检验 & 需要 Box-Cox 的属性 ----------
    normality_index = normality_test(X_tr, y_tr, n_class, n_attr)
    # 条件 3：若“非正态”类 ≥ 一半，则对该属性整体做 Box-Cox
    transform_flags = normality_index.sum(axis=0) >= (n_class / 2)

    lambdas: list[list[float | None]] = [[None] * n_attr for _ in range(n_class)]
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
    # 逐类遍历计算均值与标准差，使用 tqdm 展示进度
    for i in tqdm(range(n_class), desc="Calculating means and stds", ncols=PROGRESS_NCOLS):
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
    """生成 BBA DataFrame 并保存至 ``csv_path``"""

    n_class = len(class_names)
    n_attr = len(attr_names)

    # 对齐标签索引
    indices = np.arange(len(X_all))
    train_idx, test_idx = train_test_split(
        indices, train_size=train_ratio, stratify=y_all, random_state=0
    )
    X_tr = X_all[train_idx].copy()
    y_tr = y_all[train_idx].copy()
    X_te = X_all[test_idx].copy()
    y_te = y_all[test_idx].copy()

    # 划分训练集与测试集
    train_dataset = IrisDataset(X_tr, y_tr, attr_names, class_names)
    test_dataset = IrisDataset(X_te, y_te, attr_names, class_names)

    # 为了后续 Box-Cox，保证所有数值严格为正：若 min ≤ 0，则整体平移
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

    # 构造并打印正态性检验结果表格
    df_norm = pd.DataFrame(
        normality_index,
        index=full_class_names,
        columns=attr_names,
    )

    print("\nTraining Set Normality Indices (1 表示拒绝正态假设):")
    print(df_norm.to_string())
    print("\n需 Box-Cox 变换的属性:", [attr_names[j] for j, f in enumerate(transform_flags) if f])

    # 生成 DataFrame 并写入 CSV
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

    # 按照 sample_index、属性顺序 (SL, SW, PL, PW) 排序，确保 1-150 顺序写入
    attr_order = {attr: i for i, attr in enumerate(attr_names)}
    df_out["_attr_order"] = df_out["attribute"].map(attr_order)
    df_out = df_out.sort_values(by=["sample_index", "_attr_order"]).reset_index(drop=True)
    df_out = df_out.drop(columns="_attr_order")

    ensure_dir(csv_path)
    df_out.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n已保存格式化后的 BBA 至 {csv_path.resolve()}  (共 {len(df_out)} 行)\n")
    return df_out


if __name__ == '__main__':
    X_all, y_all, attr_names, class_names, full_class_names = load_iris_data()
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

    # 生成 BBA，是最重要的函数。
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

    # 调用抽样预览函数
    preview_sample(df_out, attr_names)
