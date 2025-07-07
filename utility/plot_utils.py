# -*- coding: utf-8 -*-
"""绘图实用函数"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["highlight_overlapping_lines", "savefig"]


def highlight_overlapping_lines(ax: plt.Axes, *, tol: float = 1e-8, extra_width: float = 0.5) -> None:
    """检测并加粗完全重合的曲线

    Parameters
    ----------
    ax : plt.Axes
        需要检查的坐标轴对象。
    tol : float, optional
        判定重合的绝对容差。
    extra_width : float, optional
        重合后增加的线宽值。
    """
    lines = ax.get_lines()
    n = len(lines)
    for i in range(n):
        for j in range(i + 1, n):
            x1 = lines[i].get_xdata()
            y1 = lines[i].get_ydata()
            x2 = lines[j].get_xdata()
            y2 = lines[j].get_ydata()
            if np.allclose(x1, x2, atol=tol, rtol=0.0) and np.allclose(y1, y2, atol=tol, rtol=0.0):
                new_width = max(lines[i].get_linewidth(), lines[j].get_linewidth()) + extra_width
                lines[i].set_linewidth(new_width)
                lines[j].set_linewidth(new_width)


def savefig(fig_or_path: str | plt.Figure, path: str | None = None, *, show: bool = True) -> None:
    """统一保存图像，可选是否显示"""
    if isinstance(fig_or_path, plt.Figure):
        fig = fig_or_path
        out = path
    else:
        fig = plt.gcf()
        out = fig_or_path
    # 某些情况下三维坐标轴无法很好地应用 tight_layout，因此这里忽略相关警告以避免屏幕输出过多提示。
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*Tight layout not applied.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*figure layout has changed.*",
            category=UserWarning,
        )
        fig.tight_layout()
    if out is not None:
        fig.savefig(out)
    else:
        fig.savefig("figure.png")
    if show:
        plt.show()
