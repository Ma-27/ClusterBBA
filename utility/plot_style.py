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

# MATLAB 默认颜色循环 (用于 α = 1/10, 1/9, …, 1/2, 1, 2, …, 10) — 深色对应大α
alpha_colors = [
    '#9edae5',  # α = 1/10, light cyan
    '#dbdb8d',  # α = 1/9, pale yellow-green
    '#f7b6d2',  # α = 1/8, light pink
    '#c49c94',  # α = 1/7, light brown
    '#c5b0d5',  # α = 1/6, light purple
    '#ff9896',  # α = 1/5, light red
    '#98df8a',  # α = 1/4, light green
    '#ffbb78',  # α = 1/3, light orange
    '#aec7e8',  # α = 1/2, light blue
    '#000000',  # α = 1, black
    '#17becf',  # α = 2, cyan
    '#bcbd22',  # α = 3, yellow-green
    '#e377c2',  # α = 4, pink
    '#8c564b',  # α = 5, brown
    '#9467bd',  # α = 6, purple
    '#d62728',  # α = 7, red
    '#2ca02c',  # α = 8, green
    '#ff7f0e',  # α = 9, orange
    '#1f77b4'  # α = 10, muted blue
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
