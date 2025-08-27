# -*- coding: utf-8 -*-
"""Iris 数据集 5 折交叉验证版 BBA 生成器
====================================

基于 :mod:`xu_iris` 中实现的流程，将 ``Iris`` 数据集划分为 ``K`` 折 (默认为:data:`config.K_FOLD_SPLITS`=5)。在每个 ``test`` 折上顺时针取下一折作为``validation``，其余三折用于训练模型。随后分别对 ``validation`` 与 ``test`` 样本生成 BBA，保存至 ``kfold_xu_val_bba_iris.csv`` 与 ``kfold_xu_bba_iris.csv``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# 确保包导入路径指向项目根目录
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# 依赖本项目内现成工具函数 / 模块
from config import K_FOLD_SPLITS

# 默认随机种子，可通过命令行覆盖
DEFAULT_RANDOM_STATE = 999
from data.bba_generation.xu_iris import (
    IrisDataset,
    generate_bba_dataframe,
    ensure_dir,
    load_iris_data,
    compute_offsets,
    fit_parameters,
)

# 输出 CSV 文件保存到 data/bba_generation 同级目录
CSV_PATH_TEST = Path(__file__).resolve().parents[1] / "kfold_xu_bba_iris.csv"
CSV_PATH_VAL = Path(__file__).resolve().parents[1] / "kfold_xu_val_bba_iris.csv"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="生成 Iris 数据集 K 折交叉验证版 BBA")
    parser.add_argument(
        "--random_state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="StratifiedKFold 的随机种子",
    )
    args = parser.parse_args()

    # ---------- Step-0: 读取基础数据 ----------
    X_all, y_all, attr_names, class_names, full_class_names = load_iris_data()
    n_class = len(class_names)
    n_attr = len(attr_names)

    # 创建分层 K 折迭代器, 保证各折类别分布一致 todo 这个种子对最后的结果影响蛮大的...
    skf = StratifiedKFold(
        n_splits=K_FOLD_SPLITS, shuffle=True, random_state=args.random_state)
    # skf.split 返回 (train_idx, test_idx); 此处仅保存每个折作为测试集时的索引列表
    fold_indices = [test_idx for _, test_idx in skf.split(X_all, y_all)]

    # 每个折生成的 BBA DataFrame 将存入列表, 最后再合并
    val_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []

    for test_fold in range(K_FOLD_SPLITS):
        # 当前循环的 test_fold 作为测试集，下一折 val_fold 用作验证集
        val_fold = (test_fold + 1) % K_FOLD_SPLITS
        train_folds = [f for f in range(K_FOLD_SPLITS) if f not in {test_fold, val_fold}]

        # 根据折号拼接得到训练、验证、测试集的样本索引
        train_idx = np.concatenate([fold_indices[f] for f in train_folds])
        val_idx = fold_indices[val_fold]
        test_idx = fold_indices[test_fold]

        # ---------- Step-1: 划分训练/验证/测试集 ----------
        X_tr = X_all[train_idx].copy()  # 训练集特征
        y_tr = y_all[train_idx].copy()  # 训练集标签
        X_val = X_all[val_idx].copy()  # 验证集特征
        y_val = y_all[val_idx].copy()  # 验证集标签
        X_te = X_all[test_idx].copy()  # 测试集特征
        y_te = y_all[test_idx].copy()  # 测试集标签

        # 包装成自定义 Dataset，便于后续抽样访问
        train_dataset = IrisDataset(X_tr, y_tr, attr_names, class_names)
        val_dataset = IrisDataset(X_val, y_val, attr_names, class_names)
        test_dataset = IrisDataset(X_te, y_te, attr_names, class_names)

        # Iris 属性均为正，仍以训练集最小值校正，确保 Box-Cox 有效
        offsets = compute_offsets(X_tr)  # 计算各属性的平移量
        X_tr += offsets
        X_val += offsets
        X_te += offsets

        (
            transform_flags,
            lambdas,
            mus,
            sigmas,
            mean_vectors,
            _,
        ) = fit_parameters(X_tr, y_tr, n_class, n_attr)

        # ---------- Step-5: 调用原函数生成 BBA DataFrame （验证集 BBA） ----------
        order_val = list(train_idx) + list(val_idx)  # 先训练后验证以复用参数
        df_val = generate_bba_dataframe(
            X_all,
            y_all,
            X_tr,
            y_tr,
            train_dataset,
            val_dataset,
            attr_names,
            class_names,
            full_class_names,
            transform_flags,
            lambdas,
            mus,
            sigmas,
            mean_vectors,
            sample_indices=order_val,
            offsets=offsets,
            decimals=4,
        )
        df_val = df_val[df_val["sample_index"].isin(val_idx + 1)].copy()  # 仅保留验证集样本
        df_val["dataset_split"] = "validation"
        df_val["fold"] = val_fold

        cols = df_val.columns.tolist()
        meta_cols = [
            "sample_index",
            "ground_truth",
            "dataset_split",
            "fold",
            "attribute",
            "attribute_data",
        ]
        rest_cols = [c for c in cols if c not in meta_cols]
        df_val = df_val[meta_cols + rest_cols]

        # 按 sample_index、属性顺序排序
        attr_order = {attr: i for i, attr in enumerate(attr_names)}
        df_val["_attr_order"] = df_val["attribute"].map(attr_order)
        df_val = df_val.sort_values(by=["sample_index", "_attr_order"]).reset_index(drop=True)
        df_val = df_val.drop(columns="_attr_order")
        val_frames.append(df_val)

        # ---------- 测试集 BBA ----------
        order_test = list(train_idx) + list(test_idx)  # 先训练再测试
        df_te = generate_bba_dataframe(
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
            sample_indices=order_test,
            offsets=offsets,
            decimals=4,
        )
        df_te = df_te[df_te["sample_index"].isin(test_idx + 1)].copy()  # 仅保留测试集样本
        df_te["dataset_split"] = "test"
        df_te["fold"] = test_fold

        cols = df_te.columns.tolist()
        rest_cols = [c for c in cols if c not in meta_cols]
        df_te = df_te[meta_cols + rest_cols]

        df_te["_attr_order"] = df_te["attribute"].map(attr_order)
        df_te = df_te.sort_values(by=["sample_index", "_attr_order"]).reset_index(drop=True)
        df_te = df_te.drop(columns="_attr_order")
        test_frames.append(df_te)

    # ---------- Step-6: 合并并写入 CSV ----------
    df_val_all = pd.concat(val_frames, ignore_index=True)  # 汇总所有验证折结果
    counts = df_val_all["sample_index"].value_counts()
    if len(counts) != len(X_all) or not (counts == n_attr).all():
        raise AssertionError("验证集: 每个样本应恰好出现一次且拥有完整属性")
    df_val_all["_attr_order"] = df_val_all["attribute"].map(attr_order)
    df_val_all = df_val_all.sort_values(by=["sample_index", "_attr_order"]).reset_index(drop=True)
    df_val_all = df_val_all.drop(columns="_attr_order")
    df_val_all = df_val_all[meta_cols + rest_cols]
    ensure_dir(CSV_PATH_VAL)
    df_val_all.to_csv(CSV_PATH_VAL, index=False, encoding="utf-8")

    df_test_all = pd.concat(test_frames, ignore_index=True)  # 汇总所有测试折结果
    counts = df_test_all["sample_index"].value_counts()
    if len(counts) != len(X_all) or not (counts == n_attr).all():
        raise AssertionError("测试集: 每个样本应恰好出现一次且拥有完整属性")
    df_test_all["_attr_order"] = df_test_all["attribute"].map(attr_order)
    df_test_all = df_test_all.sort_values(by=["sample_index", "_attr_order"]).reset_index(drop=True)
    df_test_all = df_test_all.drop(columns="_attr_order")
    df_test_all = df_test_all[meta_cols + rest_cols]
    ensure_dir(CSV_PATH_TEST)
    df_test_all.to_csv(CSV_PATH_TEST, index=False, encoding="utf-8")  # 保存测试集 BBA

    print(f"\n已保存 {K_FOLD_SPLITS} 折验证 BBA 至 {CSV_PATH_VAL.resolve()}")
    print(
        f"已保存 {K_FOLD_SPLITS} 折测试 BBA 至 {CSV_PATH_TEST.resolve()}  (共 {len(df_test_all)} 行)\n"
    )
