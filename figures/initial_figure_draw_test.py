import matplotlib.pyplot as plt
import numpy as np

from utility.plot_style import apply_style

apply_style()


# 定义自造函数，用于生成各子图的大致曲线形状
def func_a1(x):
    return 0.55 * np.exp(-0.4 * x) + 0.05


def func_a2(x):
    return 0.45 * np.exp(-0.6 * x) + 0.02


def func_b1(x):
    return 0.42 * np.exp(-0.5 * x) + 0.05


def func_b2(x):
    return 0.18 + 0.02 * np.sin(0.5 * x)


def func_b3(x):
    return 0.15 * np.ones_like(x)


def func_c1(x):
    return 0.25 * (1 - np.cos(0.3 * x)) + 0.02


def func_c2(x):
    return 0.5 * (x / 10) ** 2 + 0.02


def func_c3(x):
    return 0.10 * np.ones_like(x)


def func_d1(x):
    return 0.48 * (1 - (x - 5) ** 2 / 25) + 0.02


def func_d2(x):
    return 0.10 + 0.02 * x


def func_d3(x):
    return 0.05 + 0.02 * np.sin(0.4 * x)


def func_e1(x):
    return 0.30 * np.exp(-0.2 * x) + 0.05


def func_e2(x):
    return 0.20 + 0.02 * x


def func_e3(x):
    return 0.25 * (1 - x / 10) + 0.02


def func_f1(x):
    return 0.20 + 0.01 * np.sin(0.6 * x)


def func_f2(x):
    return 0.22 * np.exp(-0.3 * x) + 0.02


def func_f3(x):
    return 0.12 + 0.02 * x


if __name__ == '__main__':
    # x 轴范围
    i = np.linspace(1, 10, 500)

    # 创建 2 行 3 列子图
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), dpi=600)

    # 子图 (a)
    ax = axes[0, 0]
    y1, y2 = func_a1(i), func_a2(i)
    ax.plot(i, y1, color='royalblue', lw=2, label=r'FBJS($m_1\Vert m_2$)')
    ax.plot(i, y2, color='darkmagenta', lw=2, label=r'FBJS($m_1\Vert m_1\oplus m_2$)')
    ax.fill_between(i, y1, y2, where=y1 >= y2, color='royalblue', alpha=0.3)
    ax.fill_between(i, y1, y2, where=y2 >= y1, color='darkmagenta', alpha=0.3)
    ax.set_xlabel('Value of $i$')
    ax.set_ylabel('Divergence measurement')
    ax.legend()
    ax.set_title('(a) CRD 后 FBJS')
    ax.grid(True, linestyle='--', linewidth=0.5)

    # 子图 (b)
    ax = axes[0, 1]
    y1, y2, y3 = func_b1(i), func_b2(i), func_b3(i)
    ax.plot(i, y1, color='royalblue', lw=2, label=r'FBJS($m_1\Vert m_2$)')
    ax.plot(i, y2, color='darkmagenta', lw=2, label=r'FBJS($m_1\Vert m_{1\cup2}$)')
    ax.plot(i, y3, color='salmon', lw=2, label=r'FBJS($m_2\Vert m_{1\cup2}$)')
    ax.fill_between(i, y1, y2, color='royalblue', alpha=0.3)
    ax.fill_between(i, y2, y3, color='darkmagenta', alpha=0.3)
    ax.set_xlabel('Value of $i$')
    ax.set_ylabel('Divergence measurement')
    ax.legend()
    ax.set_title('(b) DCR 后 FBJS')
    ax.grid(True, linestyle='--', linewidth=0.5)

    # 子图 (c)
    ax = axes[0, 2]
    y1, y2, y3 = func_c1(i), func_c2(i), func_c3(i)
    ax.plot(i, y1, color='royalblue', lw=2, label=r'BJS($m_1\Vert m_2$)')
    ax.plot(i, y2, color='darkmagenta', lw=2, label=r'BJS($m_1\Vert m_1\oplus m_2$)')
    ax.plot(i, y3, color='salmon', lw=2, label=r'BJS($m_2\Vert m_1\oplus m_2$)')
    ax.fill_between(i, y1, y2, color='royalblue', alpha=0.3)
    ax.fill_between(i, y2, y3, color='darkmagenta', alpha=0.3)
    ax.set_xlabel('Value of $i$')
    ax.set_ylabel('Divergence measurement')
    ax.legend()
    ax.set_title('(c) CRD 后 BJS')
    ax.grid(True, linestyle='--', linewidth=0.5)

    # 子图 (d)
    ax = axes[1, 0]
    y1, y2, y3 = func_d1(i), func_d2(i), func_d3(i)
    ax.plot(i, y1, color='royalblue', lw=2, label=r'BJS($m_1\Vert m_2$)')
    ax.plot(i, y2, color='darkmagenta', lw=2, label=r'BJS($m_1\Vert m_{1\cup2}$)')
    ax.plot(i, y3, color='salmon', lw=2, label=r'BJS($m_2\Vert m_{1\cup2}$)')
    ax.fill_between(i, y1, y2, color='royalblue', alpha=0.3)
    ax.fill_between(i, y2, y3, color='salmon', alpha=0.3)
    ax.set_xlabel('Value of $i$')
    ax.set_ylabel('Divergence measurement')
    ax.legend()
    ax.set_title('(d) DCR 后 BJS')
    ax.grid(True, linestyle='--', linewidth=0.5)

    # 子图 (e)
    ax = axes[1, 1]
    y1, y2, y3 = func_e1(i), func_e2(i), func_e3(i)
    ax.plot(i, y1, color='royalblue', lw=2, label=r'RB($m_1\Vert m_2$)')
    ax.plot(i, y2, color='darkmagenta', lw=2, label=r'RB($m_1\Vert m_1\oplus m_2$)')
    ax.plot(i, y3, color='salmon', lw=2, label=r'RB($m_2\Vert m_1\oplus m_2$)')
    ax.fill_between(i, y1, y2, color='darkmagenta', alpha=0.3)
    ax.fill_between(i, y2, y3, color='salmon', alpha=0.3)
    ax.set_xlabel('Value of $i$')
    ax.set_ylabel('Divergence measurement')
    ax.legend()
    ax.set_title('(e) CRD 后 RB')
    ax.grid(True, linestyle='--', linewidth=0.5)

    # 子图 (f)
    ax = axes[1, 2]
    y1, y2, y3 = func_f1(i), func_f2(i), func_f3(i)
    ax.plot(i, y1, color='royalblue', lw=2, label=r'RB($m_1\Vert m_2$)')
    ax.plot(i, y2, color='darkmagenta', lw=2, label=r'RB($m_1\Vert m_{1\cup2}$)')
    ax.plot(i, y3, color='salmon', lw=2, label=r'RB($m_2\Vert m_{1\cup2}$)')
    ax.fill_between(i, y1, y2, color='royalblue', alpha=0.3)
    ax.fill_between(i, y2, y3, color='salmon', alpha=0.3)
    ax.set_xlabel('Value of $i$')
    ax.set_ylabel('Divergence measurement')
    ax.legend()
    ax.set_title('(f) DCR 后 RB')
    ax.grid(True, linestyle='--', linewidth=0.5)

    # 布局调整并显示
    plt.tight_layout()
    plt.show()
