"""DS+SVM 基线的命令行入口。"""

import argparse
import os
import sys
import time
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# 将项目根目录加入模块搜索路径，方便直接运行该脚本
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config import K_FOLD_SPLITS
from baseline.ds_svm.dataset import load_dataset
from baseline.ds_svm.trainer import cross_validate
from experiments.application_utils import print_evaluation_matrix

# 控制 pandas 输出格式，避免科学计数法
pd.options.display.float_format = "{:.4f}".format


def evaluate_on_dataset(name: str, method: str = "multinomial", n_splits: int = K_FOLD_SPLITS, random_state: int = 42,
                        mc_M: int = 400, n_theta_grid: int = 801, ) -> Tuple[
    float, float, float, float, np.ndarray, np.ndarray, List[str]]:
    """在指定数据集上运行所选方法并返回评估结果。"""

    # 加载数据集，返回特征矩阵 X、整型标签 y 及类别名称
    X, y, class_names = load_dataset(name)
    # 调用交叉验证函数，获得各折的平均准确率与 F1 等指标
    acc_m, acc_s, f1_m, f1_s, y_true, y_pred = cross_validate(
        X, y, n_splits, random_state, method, mc_M, n_theta_grid
    )
    return acc_m, acc_s, f1_m, f1_s, y_true, y_pred, class_names


def summarize_results(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], method: str) -> Tuple[
    str, float, float, pd.DataFrame, pd.DataFrame]:
    """生成分类报告、准确率、F1 分数、摘要表和混淆矩阵。"""

    # 构造从 1 开始的类别标签列表，便于生成报告
    labels = list(range(1, len(class_names) + 1))
    # 生成详细的分类报告字符串
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names, digits=4, zero_division=0, )
    # 计算总体准确率与宏平均 F1
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro")
    # 再次生成字典形式的报告以提取每类 F1 值
    report_dict = classification_report(y_true, y_pred, labels=labels, target_names=class_names, output_dict=True,
                                        zero_division=0, )
    class_f1 = [round(report_dict[name]["f1-score"], 4) for name in class_names]
    # 汇总为表格，便于与其他方法比较
    summary = pd.DataFrame({
        "Class / Metric": class_names + ["Accuracy", "F1score"],
        method: class_f1 + [round(acc, 4), round(f1_macro, 4)],
    })
    # 计算并封装混淆矩阵，行列均为类别名称
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    return report, acc, f1_macro, summary, cm_df


if __name__ == "__main__":
    # 构建命令行参数解析器
    parser = argparse.ArgumentParser(description="DS + SVM 证据化基线")
    # todo 可以选择单分类或者是多分类方法
    parser.add_argument("--method", choices=["logistic", "multinomial"], default="multinomial",
                        help="选择使用的校准方法", )
    # todo 选取评估的数据集
    parser.add_argument("--dataset", default="iris", help="需要评估的数据集", )
    parser.add_argument("--splits", type=int, default=K_FOLD_SPLITS, help="交叉验证折数")
    parser.add_argument("--mc_M", type=int, default=400, help="多项式方法的 Monte Carlo 次数")
    parser.add_argument("--theta_grid", type=int, default=801, help="多项式方法的 θ 网格大小")
    args = parser.parse_args()

    # 忽略一些外部库的无关警告
    warnings.filterwarnings("ignore")

    # 记录开始时间以计算训练耗时
    start_time = time.time()
    try:
        acc_m, acc_s, f1_m, f1_s, y_true, y_pred, class_names = evaluate_on_dataset(
            args.dataset,
            method=args.method,
            n_splits=args.splits,
            random_state=42,
            mc_M=args.mc_M,
            n_theta_grid=args.theta_grid,
        )
    except Exception as exc:  # pragma: no cover
        # 捕获异常并提示
        print(f"[WARN] {args.dataset} evaluation failed: {exc}")
        sys.exit(1)

    time_elapsed = time.time() - start_time
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print(f"Best val Acc: {acc_m:.4f}")

    # 汇总并打印各类评估结果
    report, acc, f1, summary, cm_df = summarize_results(
        y_true, y_pred, class_names, args.method
    )
    print(report)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 score: {f1:.4f}")
    print(summary.to_markdown(index=False, floatfmt=".4f"))
    print("\nConfusion Matrix:")
    print(cm_df.to_string())
    # 追加打印包含 TP、FP 等统计量的评估矩阵
    print("\nAdditional Evaluation Metrics:")
    print_evaluation_matrix(y_true.tolist(), y_pred.tolist(), args.method)
