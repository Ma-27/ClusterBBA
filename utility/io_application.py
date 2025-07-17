# -*- coding: utf-8 -*-
"""应用数据集 I/O
=================

读取 ``kfold_xu_bba_iris.csv``，该文件按以下列组织：

``sample_index``、``ground_truth``、``dataset_split``、``fold``（可选）、``attribute``、``attribute_data`` 以及若干焦元质量列 ``{Vi}``、``{Ve}`` 等。每个样本包含多行，对应其不同属性的 BBA。
本模块按 ``sample_index`` 分组，返回每个样本的 BBA 列表及其真实标签。该 CSV 路径由 :data:`CSV_PATH` 给出，外部无需再关心具体位置。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm

# 依赖本项目内现成工具函数 / 模块
from config import PROGRESS_NCOLS
from utility.bba import BBA

# 默认路径名称
CSV_PATH = Path(__file__).resolve().parents[1] / "data" / "kfold_xu_bba_iris.csv"

__all__ = ["load_application_dataset"]


def load_application_dataset(csv_path: str | Path = CSV_PATH, *, debug: bool = True,
                             debug_samples: int = 2
                             ) -> List[Tuple[int, List[BBA], str]]:
    """读取并解析应用数据集.

    Parameters
    ----------
    csv_path : str or Path, optional
        CSV 文件路径，默认为 :data:`CSV_PATH`。
    debug : bool, optional
        是否调试模式，仅读取前 ``debug_samples`` 个样本，默认为 ``True``。
    debug_samples : int, optional
        调试模式下读取的样本数量，默认为 ``2``。

    Returns
    -------
    list
        ``[(index, [BBA, ...], ground_truth), ...]``
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"找不到数据文件: {csv_path}")

    df = pd.read_csv(csv_path)

    # 只保留质量值列，其余列都是元数据
    meta_cols = [
        "sample_index",
        "ground_truth",
        "dataset_split",
        "attribute",
        "attribute_data",
    ]
    if "fold" in df.columns:
        meta_cols.append("fold")
    bba_cols = [c for c in df.columns if c not in meta_cols]
    # 从列名解析所有出现的焦元，构造统一的框架
    frame = set()
    for col in bba_cols:
        frame.update(BBA.parse_focal_set(col))

    samples: List[Tuple[int, List[BBA], str]] = []
    # ``sample_index`` 将同一样本的多行 BBA 归为一组
    total = df["sample_index"].nunique()
    if debug:
        total = min(total, debug_samples)
    # 用 tqdm 展示读取进度，防止加载过程无响应
    group_iter = tqdm(
        df.groupby("sample_index"),
        desc="读取数据",
        total=total,
        ncols=PROGRESS_NCOLS,
    )
    for idx, group in group_iter:
        bbas: List[BBA] = []
        # 每行对应一个属性的 BBA
        for _, row in group.iterrows():
            # 将一行的各质量值解析成 mass 字典
            mass = {BBA.parse_focal_set(col): float(row[col]) for col in bba_cols}
            name = f"m_{row['attribute']}"  # 以属性名作为 BBA 名称
            bbas.append(BBA(mass, frame=frame, name=name))
        # 所有行共享同一标签，直接取第一行
        gt = str(group.iloc[0]["ground_truth"])
        samples.append((int(idx), bbas, gt))
        if debug and len(samples) >= debug_samples:
            # 调试模式下仅保留前若干个样本
            break

    return samples
