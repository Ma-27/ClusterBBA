# -*- coding: utf-8 -*-
"""test_xu_iris.py

验证生成的 ``Iris`` BBA 与原始数据的一致性。

本文件同时测试 ``xu_iris.py`` 与 ``kfold_xu_iris.py`` 生成的 CSV：

1. 所有样本是否完整且无重复；
2. 每行 BBA 质量是否归一化；
3. ``kfold`` 版本的 ``sample_index`` 是否与原始数据索引一致；
4. 给定特征组合 ``(5.0, 3.6, 1.4, 0.2)`` 在两个 CSV 中对应的 ``sample_index``。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

# 确保包导入路径指向项目根目录
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# 依赖本项目内现成工具函数 / 模块
from data.bba_generation.kfold_xu_iris import (
    CSV_PATH as KFOLD_CSV_PATH,
    load_iris_data,
)
from data.bba_generation.xu_iris import CSV_PATH as BASIC_CSV_PATH
import numpy as np


def _print_md_sample(df: pd.DataFrame, n: int = 5) -> None:
    """以 Markdown 表格格式打印给定 ``DataFrame`` 前 ``n`` 行样例。"""

    print(df.head(n).to_markdown(index=False))


def test_index_mapping() -> None:
    """检查 ``kfold`` CSV 的 ``sample_index`` 映射。"""
    print("开始测试 sample_index 映射……")
    if not KFOLD_CSV_PATH.exists():
        subprocess.run(["python", "-m", "data.bba_generation.kfold_xu_iris"], check=True)
    df = pd.read_csv(KFOLD_CSV_PATH)
    # 有些环境可能将 ``sample_index`` 解析为浮点数，这里统一转为 ``int``
    df["sample_index"] = df["sample_index"].astype(int)
    X_all, _, attr_names, _, _ = load_iris_data()
    n_attr = len(attr_names)
    ok = True

    try:
        # 所有索引应覆盖 ``1..len(X_all)``
        assert sorted(df["sample_index"].unique()) == list(range(1, len(X_all) + 1))
        print(" - 索引范围检查通过")
    except AssertionError:
        print(" - 索引范围检查失败")
        ok = False

    counts = df["sample_index"].value_counts()
    try:
        assert (counts == n_attr).all()
        print(" - 每个样本行数检查通过")
    except AssertionError:
        print(" - 每个样本行数检查失败")
        ok = False

    if ok:
        print("sample_index 映射检查通过")
    else:
        raise AssertionError("sample_index 映射检查未通过")


def _load_dataset() -> pd.DataFrame:
    """读取原始 Iris 数据集并返回带标签的 ``DataFrame``。"""
    X_all, y_all, attr_names, _, full_class_names = load_iris_data()
    df = pd.DataFrame(X_all, columns=attr_names)
    df["ground_truth"] = [full_class_names[i] for i in y_all]
    # 与生成的 CSV 保持一致, 将索引调整为从 1 开始
    df.index = df.index + 1
    return df


def _parse_bba(df: pd.DataFrame, attr_names: list[str]) -> dict[int, tuple[float, ...]]:
    """按 ``sample_index`` 重组特征向量。"""
    features: dict[int, tuple[float, ...]] = {}
    for idx, grp in df.groupby("sample_index"):
        # ``grp`` 含 4 行属性，按 ``attr_names`` 排序还原为原始特征向量
        values = (
            grp.set_index("attribute")
            .loc[attr_names]["attribute_data"]
            .to_numpy()
        )
        features[int(idx)] = tuple(values.tolist())
    return features


def _check_unique_and_complete(
        df: pd.DataFrame, dataset_df: pd.DataFrame, attr_names: list[str], csv_name: str
) -> list[str]:
    """验证样本索引是否完整且每个样本恰有 ``len(attr_names)`` 行。"""
    errors: list[str] = []
    counts = df["sample_index"].value_counts().sort_index()
    # dataset_df 的索引从 1 开始, 直接比较即可
    missing = set(dataset_df.index) - set(counts.index)
    if missing:
        errors.append(f"{csv_name}: 缺失样本索引 {sorted(missing)}")
        print("示例缺失索引:")
        _print_md_sample(dataset_df.loc[sorted(missing)[:1]])
    wrong = counts[counts != len(attr_names)]
    if not wrong.empty:
        errors.append(
            f"{csv_name}: 以下索引的行数不等于 {len(attr_names)}: {list(wrong.index)}"
        )
        sample_idx = int(wrong.index[0])
        print(f"示例索引 {sample_idx}:")
        _print_md_sample(df[df["sample_index"] == sample_idx])
    return errors


def _check_decimal_places(csv_path: Path, columns: list[str], decimals: int = 4) -> list[str]:
    """检查 CSV 中指定列是否至多保留 ``decimals`` 位小数。"""
    import re

    errors: list[str] = []
    df_str = pd.read_csv(csv_path, dtype=str)
    df_full = pd.read_csv(csv_path)
    # 正则表达式，最多支持 decimal 位小数
    pattern = re.compile(rf"^-?\d+(?:\.\d{{1,{decimals}}})?$")

    for col in columns:
        bad_rows = df_str.index[~df_str[col].str.match(pattern)]
        if not bad_rows.empty:
            errors.append(
                f"{csv_path.name}: 列 {col} 存在超过 {decimals} 位小数的行，如 {(bad_rows + 2).tolist()[:5]}"
            )
            print(f"{csv_path.name} 列 {col} 示例:")
            _print_md_sample(df_full.iloc[bad_rows])
    return errors


def _check_bba_values(df: pd.DataFrame, csv_name: str) -> list[str]:
    """检查 BBA 数值是否非负且严格归一化。"""
    errors: list[str] = []
    mass_cols = [
        "{Vi}",
        "{Ve}",
        "{Se}",
        "{Vi ∪ Ve}",
        "{Vi ∪ Se}",
        "{Ve ∪ Se}",
        "{Vi ∪ Ve ∪ Se}",
    ]
    neg_rows = df.index[(df[mass_cols] < 0).any(axis=1)]
    if not neg_rows.empty:
        errors.append(f"{csv_name}: 存在负的 BBA 质量值于行 {neg_rows.tolist()}")
        print(f"{csv_name} 负值示例:")
        _print_md_sample(df.loc[neg_rows])
    over_rows = df.index[(df[mass_cols] > 1).any(axis=1)]
    if not over_rows.empty:
        errors.append(f"{csv_name}: 存在超过 1 的 BBA 质量值于行 {over_rows.tolist()}")
        print(f"{csv_name} 超过 1 示例:")
        _print_md_sample(df.loc[over_rows])
    sums = df[mass_cols].sum(axis=1)
    # 四舍五入到四位小数后仍应当精确为 1
    bad_sum = df.index[sums.round(4) != 1.0]
    if not bad_sum.empty:
        errors.append(
            f"{csv_name}: 有 {len(bad_sum)} 行质量和未严格等于 1，示例行 {bad_sum.tolist()[:5]}"
        )
        print(f"{csv_name} 归一化失败示例:")
        _print_md_sample(df.loc[bad_sum])
    return errors


def _check_kfold_alignment(
        df: pd.DataFrame, dataset_df: pd.DataFrame, attr_names: list[str]
) -> list[str]:
    """检查 ``kfold`` 版本是否与原始数据按索引完全对应。"""
    errors: list[str] = []
    # 将 ``sample_index`` 下的 4 行属性重组成特征向量，便于比对
    features = _parse_bba(df, attr_names)
    for idx, row in dataset_df.iterrows():
        if idx not in features:
            errors.append(f"kfold CSV: 缺失索引 {idx}")
            print("缺失索引示例:")
            _print_md_sample(dataset_df.loc[[idx]])
            continue
        vals = np.array(features[idx], dtype=float)
        expect = row[attr_names].to_numpy(dtype=float)
        if not np.allclose(vals, expect, atol=1e-6):
            errors.append(f"kfold CSV: 索引 {idx} 的特征不匹配")
            print("CSV 示例:")
            _print_md_sample(df[df["sample_index"] == idx])
            print("dataset 示例:")
            _print_md_sample(dataset_df.loc[[idx]])
        gt = df[df["sample_index"] == idx]["ground_truth"].iloc[0]
        if gt != row["ground_truth"]:
            errors.append(
                f"kfold CSV: 索引 {idx} 的 ground_truth 错误 {gt} != {row['ground_truth']}"
            )
            print("CSV ground_truth 示例:")
            _print_md_sample(df[df["sample_index"] == idx])
            print("dataset ground_truth:")
            _print_md_sample(dataset_df.loc[[idx]])
    return errors


def _check_basic_alignment(
        df: pd.DataFrame, dataset_df: pd.DataFrame, attr_names: list[str]
) -> list[str]:
    """检查普通版 CSV 中的特征集合与原始数据是否一致。"""
    errors: list[str] = []
    # 从 CSV 重组所有特征向量，与原数据集逐一对比
    csv_features = list(_parse_bba(df, attr_names).values())
    dataset_features = [
        tuple(float(row[attr]) for attr in attr_names) for _, row in dataset_df.iterrows()
    ]
    if sorted(csv_features) != sorted(dataset_features):
        diff_csv = [f for f in csv_features if f not in dataset_features][:5]
        diff_ds = [f for f in dataset_features if f not in csv_features][:5]
        errors.append(
            f"普通 CSV: 特征集合与原始数据不符，示例 csv 独有 {diff_csv}, dataset 独有 {diff_ds}"
        )
        print("特征集合差异示例:")
        _print_md_sample(pd.DataFrame(diff_csv, columns=attr_names))
    feature_to_label = {
        # 使用特征向量作为键, 快速查询其真实标签
        tuple(float(row[attr]) for attr in attr_names): row["ground_truth"]
        for _, row in dataset_df.iterrows()
    }
    for idx, grp in df.groupby("sample_index"):
        feat = tuple(
            grp.set_index("attribute").loc[attr_names]["attribute_data"].to_numpy(dtype=float)
        )
        gt = grp["ground_truth"].iloc[0]
        expect = feature_to_label.get(feat)
        if expect is None:
            errors.append(f"普通 CSV: 样本 {idx} 的特征未在原数据中找到")
            _print_md_sample(grp)
            continue
        if gt != expect:
            errors.append(f"普通 CSV: 样本 {idx} 的 ground_truth 错误 {gt} != {expect}")
            _print_md_sample(grp)
    return errors


def test_data_consistency() -> None:
    """检查两份 CSV 是否与原始数据一致且无重复。"""
    print("开始测试 CSV 与原始数据的一致性……")
    if not BASIC_CSV_PATH.exists():
        subprocess.run(["python", "-m", "data.bba_generation.xu_iris"], check=True)
    if not KFOLD_CSV_PATH.exists():
        subprocess.run(["python", "-m", "data.bba_generation.kfold_xu_iris"], check=True)

    df_basic = pd.read_csv(BASIC_CSV_PATH)
    df_kfold = pd.read_csv(KFOLD_CSV_PATH)
    # 确保索引为 ``int``，避免因类型不同导致比较失败
    df_basic["sample_index"] = df_basic["sample_index"].astype(int)
    df_kfold["sample_index"] = df_kfold["sample_index"].astype(int)

    dataset_df = _load_dataset()
    attr_names = ["SL", "SW", "PL", "PW"]

    errors: list[str] = []

    # ---------- 基础检查 ----------
    errs = _check_unique_and_complete(df_basic, dataset_df, attr_names, "普通 CSV")
    if errs:
        print(" - 普通 CSV 样本完整性检查失败")
        for e in errs:
            print("   ", e)
    else:
        print(" - 普通 CSV 样本完整性检查通过")
    errors += errs

    errs = _check_unique_and_complete(df_kfold, dataset_df, attr_names, "kfold CSV")
    if errs:
        print(" - kfold CSV 样本完整性检查失败")
        for e in errs:
            print("   ", e)
    else:
        print(" - kfold CSV 样本完整性检查通过")
    errors += errs

    errs = _check_bba_values(df_basic, "普通 CSV")
    if errs:
        print(" - 普通 CSV BBA 值检查失败")
        for e in errs:
            print("   ", e)
    else:
        print(" - 普通 CSV BBA 值检查通过")
    errors += errs

    decimal_cols = [
        "{Vi}", "{Ve}", "{Se}",
        "{Vi ∪ Ve}", "{Vi ∪ Se}", "{Ve ∪ Se}", "{Vi ∪ Ve ∪ Se}",
    ]
    errs = _check_decimal_places(BASIC_CSV_PATH, decimal_cols)
    if errs:
        print(" - 普通 CSV 小数位检查失败")
        for e in errs:
            print("   ", e)
    else:
        print(" - 普通 CSV 小数位检查通过")
    errors += errs

    errs = _check_bba_values(df_kfold, "kfold CSV")
    if errs:
        print(" - kfold CSV BBA 值检查失败")
        for e in errs:
            print("   ", e)
    else:
        print(" - kfold CSV BBA 值检查通过")
    errors += errs

    errs = _check_decimal_places(KFOLD_CSV_PATH, decimal_cols)
    if errs:
        print(" - kfold CSV 小数位检查失败")
        for e in errs:
            print("   ", e)
    else:
        print(" - kfold CSV 小数位检查通过")
    errors += errs

    # ---------- 与原数据对齐检查 ----------
    errs = _check_kfold_alignment(df_kfold, dataset_df, attr_names)
    if errs:
        print(" - kfold CSV 对齐检查失败")
        for e in errs:
            print("   ", e)
    else:
        print(" - kfold CSV 对齐检查通过")
    errors += errs

    errs = _check_basic_alignment(df_basic, dataset_df, attr_names)
    if errs:
        print(" - 普通 CSV 对齐检查失败")
        for e in errs:
            print("   ", e)
    else:
        print(" - 普通 CSV 对齐检查通过")
    errors += errs

    # ---------- 查找特定特征组对应的 sample_index ----------
    target = (5.0, 3.6, 1.4, 0.2)

    def find_index(df: pd.DataFrame) -> int:
        # 遍历所有样本, 找到与 ``target`` 完全匹配的特征向量
        for idx, grp in df.groupby("sample_index"):
            vals = (
                grp.set_index("attribute").loc[attr_names]["attribute_data"].to_numpy()
            )
            if np.allclose(vals, target):
                return int(idx)
        raise AssertionError("target sample not found")

    idx_basic = find_index(df_basic)
    idx_kfold = find_index(df_kfold)
    print("The (5.0, 3.6, 1.4, 0.2) vector index is:")
    print("basic csv index:", idx_basic)
    print("kfold csv index:", idx_kfold)
    dataset_idx = int(
        dataset_df.index[
            (dataset_df[attr_names] == pd.Series(target, index=attr_names)).all(axis=1)
        ][0]
    )
    print("dataset index:", dataset_idx)

    # 仅打印索引信息, 不视为错误

    if errors:
        print("检测到以下问题:")
        for e in errors:
            print("-", e)
        raise AssertionError("CSV 与原始数据存在不一致")
    else:
        print("数据一致性检查通过")


if __name__ == "__main__":  # pragma: no cover
    test_index_mapping()
    test_data_consistency()
