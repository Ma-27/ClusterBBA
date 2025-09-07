"""DS+SVM 基线的训练与评估工具。"""
from typing import List, Tuple

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .calibration_logistic import BinaryEvidentialLogisticCalibration
from .calibration_multinomial import EvidentialMultinomialCalibration


def train_ova_svm(X_tr_cal: np.ndarray, y_tr_cal: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                  split_seed: int = 0, ) -> Tuple[
    np.ndarray, np.ndarray, List[BinaryEvidentialLogisticCalibration], Tuple[np.ndarray, np.ndarray]]:
    """将训练集一分为二，并训练每类的 OVA-SVM 模型。"""

    # 采用分层抽样将训练集一分为二，保证每部分包含所有类别
    X_svm, X_cal, y_svm, y_cal = train_test_split(
        X_tr_cal, y_tr_cal, test_size=0.5, random_state=split_seed, stratify=y_tr_cal
    )

    K = int(np.max(y_tr_cal))
    # 预分配 OVA 得分矩阵，行数为样本数，列数为类别数
    S_cal = np.zeros((X_cal.shape[0], K), dtype=float)
    S_val = np.zeros((X_val.shape[0], K), dtype=float)
    calibrators: List[BinaryEvidentialLogisticCalibration] = []

    for k in tqdm(range(1, K + 1), desc="训练 OVA-SVM", leave=False):
        # 将当前类别视为正类训练二分类 SVM
        y_bin_svm = (y_svm == k).astype(int)
        pipe = make_pipeline(StandardScaler(), svm.SVC(kernel="rbf", C=10.0, gamma="scale"))
        pipe.fit(X_svm, y_bin_svm)
        # 得到在校准集与验证集上的决策函数分数
        s_cal = pipe.decision_function(X_cal)
        s_val = pipe.decision_function(X_val)
        # 分数写入对应的列，之后可用于证据化处理
        S_cal[:, k - 1] = s_cal
        S_val[:, k - 1] = s_val
        # 记录该类别的 Logistic 标定器以便后续使用
        y_bin_cal = (y_cal == k).astype(int)
        calibrators.append(BinaryEvidentialLogisticCalibration(s_cal=s_cal, y_cal=y_bin_cal))

    return S_cal, S_val, calibrators, (X_cal, y_cal)


def evaluate_logistic_calibration(S_val: np.ndarray, calibrators: List[BinaryEvidentialLogisticCalibration],
                                  y_true: np.ndarray, ) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """使用 Logistic 标定法评估验证集。返回 ``(accuracy, macro-F1, y_pred, pl_val)``。"""

    K = S_val.shape[1]
    # 预分配可能性矩阵 ``pl_val``，后续逐类填充
    pl_val = np.zeros((S_val.shape[0], K), dtype=float)
    # 对每个类别分别计算其可能性
    for k in tqdm(range(K), desc="评估 Logistic 校准", leave=False):
        calib = calibrators[k]
        for i in range(S_val.shape[0]):
            # 对第 i 个样本计算属于第 k 类的可能性
            pl_val[i, k] = calib.plausibility_positive(S_val[i, k])
    # 预测取可能性最大的类别，标签从 1 开始
    y_pred = 1 + np.argmax(pl_val, axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return float(acc), float(f1), y_pred, pl_val


def evaluate_multinomial_calibration(S_cal: np.ndarray, y_cal: np.ndarray, S_val: np.ndarray, y_true: np.ndarray,
                                     mc_M: int, n_theta_grid: int, ) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """使用多项式校准法评估验证集。返回 ``(accuracy, macro-F1, y_pred, pl_val)``。"""
    K = S_val.shape[1]
    # 构建多项式证据化校准器，该校准器可估计任意单例的可能性
    emc = EvidentialMultinomialCalibration(
        S_cal=S_cal, y_cal=y_cal, mc_M=mc_M, n_theta_grid=n_theta_grid
    )
    pl_val = np.zeros((S_val.shape[0], K), dtype=float)
    for i in tqdm(range(S_val.shape[0]), desc="评估多项式校准", leave=False):
        # 计算第 i 个样本属于各类别的可能性
        pl_val[i, :] = emc.pl_singleton(S_val[i, :])
    y_pred = 1 + np.argmax(pl_val, axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return float(acc), float(f1), y_pred, pl_val


def cross_validate(X: np.ndarray, y: np.ndarray, n_splits: int, random_state: int, method: str, mc_M: int,
                   n_theta_grid: int, ) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """执行交叉验证并返回性能统计以及预测结果与概率矩阵。"""

    # 初始化分层 K 折交叉验证器，保证每折类分布一致
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    acc_scores, f1_scores = [], []
    y_true_all, y_pred_all, y_score_all = [], [], []
    for fold, (idx_tr, idx_te) in enumerate(
            tqdm(skf.split(X, y), total=n_splits, desc="交叉验证", leave=False)
    ):
        # 根据索引切分训练集与测试集
        X_tr, X_te = X[idx_tr], X[idx_te]
        y_tr, y_te = y[idx_tr], y[idx_te]
        # 先训练 OVA-SVM，并得到验证集分数与标定器
        S_cal, S_val, calibrators, (X_cal, y_cal) = train_ova_svm(
            X_tr, y_tr, X_te, y_te, split_seed=fold
        )
        # 根据选择的校准方法进行评估
        if method.lower() == "logistic":
            acc, f1, y_pred, pl_val = evaluate_logistic_calibration(S_val, calibrators, y_te)
        else:
            acc, f1, y_pred, pl_val = evaluate_multinomial_calibration(S_cal, y_cal, S_val, y_te, mc_M, n_theta_grid)
        # 收集每折的分数与预测
        acc_scores.append(acc)
        f1_scores.append(f1)
        y_true_all.extend(y_te)
        y_pred_all.extend(y_pred)
        y_score_all.extend(pl_val)
    # 计算交叉验证的均值与标准差
    acc_mean = float(np.mean(acc_scores))
    acc_std = float(np.std(acc_scores))
    f1_mean = float(np.mean(f1_scores))
    f1_std = float(np.std(f1_scores))
    return (acc_mean, acc_std, f1_mean, f1_std, np.array(y_true_all), np.array(y_pred_all), np.array(y_score_all),)
