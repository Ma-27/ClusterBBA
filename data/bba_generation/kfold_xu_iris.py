# -*- coding: utf-8 -*-
"""Iris 数据集 K 折交叉验证版 BBA 生成器
====================================

本脚本基于 :mod:`xu_iris` 中实现的生成流程, 对 ``Iris`` 数据集进行``K`` 折交叉验证 (默认为 :data:`config.K_FOLD_SPLITS`=5) 。在每个折上重建 ``Box-Cox`` 变换和正态分布模型, 并按 Xu 等人提出的方法为每个样本、每个属性生成 BBA。生成的所有折被合并保存至 ``kfold_xu_bba_iris.csv``，其中额外记录 ``fold`` 序号, ``dataset_split`` 字段统一填写 ``test``；此外，每个外层折还会生成验证集 CSV，保存在 ``bba_validation/kfold_xu_bbavalset_iris`` 目录下，其 ``dataset_split`` 字段统一填写 ``validation``。可通过 ``--random_state`` 参数指定交叉验证的随机种子。
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
CSV_PATH = Path(__file__).resolve().parents[1] / "kfold_xu_bba_iris.csv"
# 验证集 CSV 保存目录，每个外层折会在此生成一个对应的验证集文件
VAL_DIR = (Path(__file__).resolve().parents[1] / "bba_validation" / "kfold_xu_bbavalset_iris")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 Iris 数据集 K 折交叉验证版 BBA")
    # 指定种子
    parser.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE, help="StratifiedKFold 的随机种子", )
    args = parser.parse_args()

    # ---------- Step-0: 读取基础数据 ----------
    X_all, y_all, attr_names, class_names, full_class_names = load_iris_data()
    n_class = len(class_names)
    n_attr = len(attr_names)

    # 创建分层 K 折迭代器, 保证各折类别分布一致 todo 这个种子对最后的结果影响蛮大的...
    skf = StratifiedKFold(
        n_splits=K_FOLD_SPLITS, shuffle=True, random_state=args.random_state)
    # 先一次性获取 5 折的索引列表, 以便外层和内层循环均复用
    fold_indices = [test_idx for _, test_idx in skf.split(X_all, y_all)]

    # 每个折生成的 BBA DataFrame 将存入此列表, 最后再合并
    fold_frames = []

    for fold in range(K_FOLD_SPLITS):
        # ---------- Step-1: 按当前折划分训练/测试集 ----------
        test_idx = fold_indices[fold]
        # 训练集由其余四折拼接而成, 保持与初始划分一致
        train_idx = np.concatenate(
            [fold_indices[i] for i in range(K_FOLD_SPLITS) if i != fold]
        )
        X_tr_raw = X_all[train_idx].copy()
        y_tr_raw = y_all[train_idx].copy()
        X_te = X_all[test_idx].copy()
        y_te = y_all[test_idx].copy()

        # --- 外层: 生成测试集 BBA ---
        X_tr = X_tr_raw.copy()
        y_tr = y_tr_raw.copy()
        train_dataset = IrisDataset(X_tr, y_tr, attr_names, class_names)
        test_dataset = IrisDataset(X_te, y_te, attr_names, class_names)

        # Iris 属性均为正，仍以训练集最小值校正，确保 Box-Cox 有效
        offsets = compute_offsets(X_tr)
        X_tr += offsets
        X_te += offsets

        (transform_flags, lambdas, mus, sigmas, mean_vectors, _,) = fit_parameters(X_tr, y_tr, n_class, n_attr)

        # ---------- Step-5: 调用原函数生成 BBA DataFrame ----------
        dataset_order = list(train_idx) + list(test_idx)
        df_fold = generate_bba_dataframe(
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
            decimals=4,
        )

        # 仅保留当前折的测试样本
        df_fold = df_fold[df_fold["sample_index"].isin(test_idx + 1)].copy()
        df_fold["dataset_split"] = "test"
        # 只保留测试样本, 训练样本仅用于建立模型
        df_fold["fold"] = fold

        # 调整列顺序: dataset_split 之后插入 fold
        cols = df_fold.columns.tolist()
        meta_cols = ["sample_index", "ground_truth", "dataset_split", "fold", "attribute", "attribute_data"]
        rest_cols = [c for c in cols if c not in meta_cols]
        df_fold = df_fold[meta_cols + rest_cols]

        # 按 sample_index、属性顺序排序
        attr_order = {attr: i for i, attr in enumerate(attr_names)}
        df_fold["_attr_order"] = df_fold["attribute"].map(attr_order)
        df_fold = df_fold.sort_values(by=["sample_index", "_attr_order"]).reset_index(drop=True)
        df_fold = df_fold.drop(columns="_attr_order")
        fold_frames.append(df_fold)

        # --- 内层: 4 折交叉验证生成验证集 BBA ---
        # 该内层循环沿用最初划分的 5 折, 去掉当前外层测试折后剩余的 4 折
        # 不重新随机划分, 直接以这 4 折轮流作为验证集
        inner_fold_indices = [
            fold_indices[i] for i in range(K_FOLD_SPLITS) if i != fold
        ]
        # 用于收集每个内层折生成的 DataFrame, 最终拼接为完整验证集
        val_frames: list[pd.DataFrame] = []
        for inner_fold in range(K_FOLD_SPLITS - 1):
            val_idx = inner_fold_indices[inner_fold]
            # 内层训练集为剩余 3 折的并集, 保持原有划分顺序
            train_inner_idx = np.concatenate(
                [inner_fold_indices[j] for j in range(K_FOLD_SPLITS - 1) if j != inner_fold]
            )
            X_in_tr = X_all[train_inner_idx].copy()
            y_in_tr = y_all[train_inner_idx].copy()
            X_in_val = X_all[val_idx].copy()
            y_in_val = y_all[val_idx].copy()

            # 包装为 IrisDataset, 复用已有生成 BBA 的 API
            train_ds_in = IrisDataset(X_in_tr, y_in_tr, attr_names, class_names)
            val_ds = IrisDataset(X_in_val, y_in_val, attr_names, class_names)

            # 为确保 Box-Cox 变换有效, 重新计算偏移量并应用于当前内折数据
            offsets_in = compute_offsets(X_in_tr)
            X_in_tr += offsets_in
            X_in_val += offsets_in

            # 拟合当前内折的 Box-Cox 与正态分布参数
            (tf_in, lam_in, mu_in, sig_in, mean_vec_in, _,) = fit_parameters(X_in_tr, y_in_tr, n_class, n_attr)

            # 确定样本在原始数据中的顺序, 便于生成 BBA 后筛选
            dataset_order_in = list(train_inner_idx) + list(val_idx)
            # 在当前内折上生成训练+验证集所有样本的 BBA
            df_val = generate_bba_dataframe(
                X_all,
                y_all,
                X_in_tr,
                y_in_tr,
                train_ds_in,
                val_ds,
                attr_names,
                class_names,
                full_class_names,
                tf_in,
                lam_in,
                mu_in,
                sig_in,
                mean_vec_in,
                sample_indices=dataset_order_in,
                offsets=offsets_in,
                decimals=4,
            )

            # 仅保留验证集中样本的 BBA, 并换算为全局样本索引 (1 起始)
            val_indices_global = val_idx + 1
            df_val = df_val[df_val["sample_index"].isin(val_indices_global)].copy()
            df_val["dataset_split"] = "validation"
            df_val["fold"] = inner_fold  # 记录内层折号

            # 调整列顺序, 保持与外层 CSV 一致
            cols = df_val.columns.tolist()
            df_val = df_val[
                ["sample_index", "ground_truth", "dataset_split", "fold", "attribute", "attribute_data", ]
                + [c for c in cols if c not in meta_cols]
                ]

            # 按样本索引及属性顺序排序, 便于后续拼接
            attr_order_in = {attr: i for i, attr in enumerate(attr_names)}
            df_val["_attr_order"] = df_val["attribute"].map(attr_order_in)
            df_val = df_val.sort_values(by=["sample_index", "_attr_order"]).reset_index(
                drop=True
            )
            df_val = df_val.drop(columns="_attr_order")
            val_frames.append(df_val)

        # 拼接四个内层折得到当前外层折对应的完整验证集
        df_val_all = pd.concat(val_frames, ignore_index=True)
        # 再次检查每个样本是否只出现一次且具有全部属性列
        counts = df_val_all["sample_index"].value_counts()
        if len(counts) != len(train_idx) or not (counts == n_attr).all():
            raise AssertionError("每个样本应恰好出现一次且拥有完整属性")
        attr_order = {attr: i for i, attr in enumerate(attr_names)}
        df_val_all["_attr_order"] = df_val_all["attribute"].map(attr_order)
        df_val_all = df_val_all.sort_values(by=["sample_index", "_attr_order"]).reset_index(
            drop=True
        )
        df_val_all = df_val_all.drop(columns="_attr_order")

        # 输出前重新整理列顺序
        cols = df_val_all.columns.tolist()
        df_val_all = df_val_all[meta_cols + [c for c in cols if c not in meta_cols]]
        val_csv_path = VAL_DIR / f"bbavalset_iris_fold{fold}.csv"
        ensure_dir(val_csv_path)  # 若目录不存在则创建
        df_val_all.to_csv(val_csv_path, index=False, encoding="utf-8")

    # ---------- Step-6: 合并并写入 CSV ----------
    df_all = pd.concat(fold_frames, ignore_index=True)
    # ---------- Step-6.1: 验证索引与行数 ----------
    counts = df_all["sample_index"].value_counts()
    if len(counts) != len(X_all) or not (counts == n_attr).all():
        raise AssertionError("每个样本应恰好出现一次且拥有完整属性")
    # 按 sample_index、属性顺序 (SL, SW, PL, PW) 排序
    attr_order = {attr: i for i, attr in enumerate(attr_names)}
    df_all["_attr_order"] = df_all["attribute"].map(attr_order)
    df_all = df_all.sort_values(by=["sample_index", "_attr_order"]).reset_index(drop=True)
    df_all = df_all.drop(columns="_attr_order")

    cols = df_all.columns.tolist()
    meta_cols = ["sample_index", "ground_truth", "dataset_split", "fold", "attribute", "attribute_data"]
    rest_cols = [c for c in cols if c not in meta_cols]
    df_all = df_all[meta_cols + rest_cols]
    ensure_dir(CSV_PATH)
    df_all.to_csv(CSV_PATH, index=False, encoding="utf-8")
    print(
        f"\n已保存 {K_FOLD_SPLITS} 折 BBA 至 {CSV_PATH.resolve()}  (共 {len(df_all)} 行)\n"
    )
