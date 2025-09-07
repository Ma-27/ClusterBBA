# -*- coding: utf-8 -*-
"""Glass 数据集 K 折交叉验证版 BBA 生成器
=======================================

基于 :mod:`xu_glass` 实现的生成流程，对 ``Glass`` 数据集进行 ``K`` 折交叉验证(默认为 :data:`config.K_FOLD_SPLITS`=5)。在每个折上重建 ``Box-Cox`` 变换和正态分布模型，按 Xu 等人的方法为每个样本、每个属性生成 BBA。所有折合并后保存至 ``kfold_xu_bba_glass.csv``，若使用 ``--use_existing_csv`` 参数，则另存为 ``kfold_xu_bba_glass_resplit.csv``，其中额外记录 ``fold`` 序号，``dataset_split`` 字段统一填写 ``test``。可通过 ``--random_state`` 参数指定交叉验证的随机种子。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

# 确保包导入路径指向项目根目录
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# 依赖本项目内现成工具函数 / 模块
from config import K_FOLD_SPLITS
import data.bba_generation.xu_glass as xu_glass
from data.bba_generation.xu_glass import (
    GlassDataset,
    generate_bba_dataframe,
    ensure_dir,
    load_glass_data,
    compute_offsets,
    fit_parameters,
    CSV_PATH as BASIC_CSV_PATH,
)

# 输出 CSV 文件保存到 data/bba_generation 同级目录
CSV_PATH = Path(__file__).resolve().parents[1] / "kfold_xu_bba_glass.csv"
# 若从已有 CSV 重新划分，则输出为 kfold_xu_bba_glass_resplit.csv
CSV_RESPLIT_PATH = CSV_PATH.with_name("kfold_xu_bba_glass_resplit.csv")
# 默认随机种子，可通过命令行覆盖
DEFAULT_RANDOM_STATE = 621

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 Glass 数据集 K 折交叉验证版 BBA")
    # 指定种子
    parser.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE, help="StratifiedKFold 的随机种子", )
    # 直接对已有的 xu_bba_glass.csv 进行 K 折划分
    parser.add_argument("--use_existing_csv", action="store_true", help="直接对已有的 xu_bba_glass.csv 进行 K 折划分", )
    args = parser.parse_args()

    # 若指定 use_existing_csv, 则直接对已有的 BASIC_CSV_PATH 进行 K 折划分
    if args.use_existing_csv:
        X_all, y_all, attr_names, _, _ = load_glass_data()
        df_all = pd.read_csv(BASIC_CSV_PATH)
        df_all["sample_index"] = df_all["sample_index"].astype(int)
        # 创建分层 K 折迭代器，保证各折类别分布一致
        skf = StratifiedKFold(n_splits=K_FOLD_SPLITS, shuffle=True, random_state=args.random_state)
        fold_map: dict[int, int] = {}
        # 生成 sample_index 到 fold 的映射
        for fold, (_, test_idx) in enumerate(skf.split(X_all, y_all)):
            for idx in test_idx:
                fold_map[idx + 1] = fold
        # 映射 sample_index 到fold
        df_all["fold"] = df_all["sample_index"].map(fold_map)
        df_all["dataset_split"] = "test"
        attr_order = {attr: i for i, attr in enumerate(attr_names)}
        df_all["_attr_order"] = df_all["attribute"].map(attr_order)

        # 按 sample_index、属性顺序 (RI, Na, Mg, ...) 排序
        df_all = (
            df_all.sort_values(by=["sample_index", "_attr_order"]) \
                .reset_index(drop=True)
        )

        # 删除辅助列
        df_all = df_all.drop(columns="_attr_order")
        cols = df_all.columns.tolist()

        # 调整列顺序: dataset_split 之后插入 fold
        meta_cols = [
            "sample_index",
            "ground_truth",
            "dataset_split",
            "fold",
            "attribute",
            "attribute_data",
        ]
        # 确保 meta_cols 顺序正确且不重复
        rest_cols = [c for c in cols if c not in meta_cols]
        df_all = df_all[meta_cols + rest_cols]

        ensure_dir(CSV_RESPLIT_PATH)
        df_all.to_csv(CSV_RESPLIT_PATH, index=False, encoding="utf-8")
        print(f"\n已保存 {K_FOLD_SPLITS} 折 BBA 至 {CSV_RESPLIT_PATH.resolve()}  (共 {len(df_all)} 行)\n")
        sys.exit(0)

    # ---------- Step-0: 读取基础数据 ----------
    X_all, y_all, attr_names, class_names, full_class_names = load_glass_data()
    n_class = len(class_names)
    n_attr = len(attr_names)
    # 由于 :func:`xu_glass.fit_parameters` 会在内部打印属性名，
    # 需将其暴露为模块级变量以供访问
    xu_glass.attr_names = attr_names

    # 创建分层 K 折迭代器，保证各折类别分布一致
    skf = StratifiedKFold(
        n_splits=K_FOLD_SPLITS, shuffle=True, random_state=args.random_state)
    # 每个折生成的 BBA DataFrame 将存入此列表，最后再合并
    fold_frames: list[pd.DataFrame] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all)):
        # ---------- Step-1: 划分当前折训练/测试集 ----------
        X_tr = X_all[train_idx].copy()
        y_tr = y_all[train_idx].copy()
        X_te = X_all[test_idx].copy()
        y_te = y_all[test_idx].copy()

        # 包装成自定义 Dataset，便于后续抽样访问
        train_dataset = GlassDataset(X_tr, y_tr, attr_names, class_names)
        test_dataset = GlassDataset(X_te, y_te, attr_names, class_names)

        # Glass 属性中可能存在非正值，同样以训练集最小值校正，确保 Box-Cox 有效
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
        # dataset_split 在交叉验证中统一标记为 test
        df_fold["dataset_split"] = "test"
        # 只保留测试样本，训练样本仅用于建立模型
        df_fold["fold"] = fold

        # 调整列顺序: dataset_split 之后插入 fold
        cols = df_fold.columns.tolist()
        meta_cols = ["sample_index", "ground_truth", "dataset_split", "fold", "attribute", "attribute_data"]
        rest_cols = [c for c in cols if c not in meta_cols]
        df_fold = df_fold[meta_cols + rest_cols]

        # 按 sample_index、属性顺序排序，确保输出有序
        attr_order = {attr: i for i, attr in enumerate(attr_names)}
        df_fold["_attr_order"] = df_fold["attribute"].map(attr_order)
        df_fold = df_fold.sort_values(by=["sample_index", "_attr_order"]).reset_index(drop=True)
        df_fold = df_fold.drop(columns="_attr_order")
        fold_frames.append(df_fold)

    # ---------- Step-6: 合并并写入 CSV ----------
    df_all = pd.concat(fold_frames, ignore_index=True)
    # ---------- Step-6.1: 验证索引与行数 ----------
    counts = df_all["sample_index"].value_counts()
    if len(counts) != len(X_all) or not (counts == n_attr).all():
        raise AssertionError("每个样本应恰好出现一次且拥有完整属性")

    # 按 sample_index、属性顺序 (RI, Na, Mg, ...) 排序
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
