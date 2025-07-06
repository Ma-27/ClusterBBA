# -*- coding: utf-8 -*-
"""统一 Matplotlib 绘图风格的辅助模块"""

from __future__ import annotations

import matplotlib.pyplot as plt
from cycler import cycler

__all__ = ["apply_style"]

# 图例配色方案
dark2_8 = [
    '#1B9E77',  # 深绿
    '#D95F02',  # 橙
    '#7570B3',  # 紫
    '#E7298A',  # 粉
    '#66A61E',  # 亮绿
    '#E6AB02',  # 黄
    '#A6761D',  # 棕
    '#666666',  # 中灰
]


def apply_style() -> None:
    """应用 SCI 论文级别的绘图默认参数"""
    plt.rcParams.update({
        # 画布和分辨率
        'figure.figsize': (8.5 / 2.54, 6 / 2.54),
        'figure.dpi': 800,
        'savefig.dpi': 800,

        # 颜色和线条
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 5,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 6,

        # 线条、标记与颜色
        'lines.linewidth': 0.6,
        'lines.markersize': 4,
        'axes.prop_cycle': cycler('color', dark2_8),

        # 网格线样式
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.6,

        # 图例样式
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.framealpha': 1.0,

        # 坐标轴与刻度样式
        'axes.linewidth': 0.8,  # 坐标轴线宽
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.minor.visible': True,
        'xtick.major.width': 0.8,  # 主刻度线宽
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.5,  # 副刻度线宽
        'ytick.minor.width': 0.5,  # 副刻度线宽
    })
