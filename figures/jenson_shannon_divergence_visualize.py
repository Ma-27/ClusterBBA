import matplotlib.pyplot as plt
import numpy as np


# 定义 KL 散度函数
def kl_divergence(p, q):
    p = np.array(p)
    q = np.array(q)
    mask = (p > 0)
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))


# 定义加权 JS 散度函数
def weighted_js_divergence(p, q, alpha):
    m = alpha * p + (1 - alpha) * q
    return alpha * kl_divergence(p, m) + (1 - alpha) * kl_divergence(q, m)


if __name__ == '__main__':
    # 网格生成
    grid_size = 100
    p_vals = np.linspace(0, 1, grid_size)
    q_vals = np.linspace(0, 1, grid_size)
    P, Q = np.meshgrid(p_vals, q_vals)

    # 选择 alpha 权重
    alpha = 0.0001

    # 计算加权 JS 距离（metric = sqrt(weighted JS divergence)）
    WJS_dist = np.zeros_like(P)
    for i in range(grid_size):
        for j in range(grid_size):
            p = np.array([P[i, j], 1 - P[i, j]])
            q = np.array([Q[i, j], 1 - Q[i, j]])
            val = weighted_js_divergence(p, q, alpha)
            WJS_dist[i, j] = np.sqrt(val) if val >= 0 else np.nan

    # 绘制三维曲面图，换个视角
    fig = plt.figure(figsize=(8, 6), dpi=600)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(P, Q, WJS_dist, rstride=3, cstride=3, edgecolor='w', linewidth=0.3)
    ax.set_xlabel('p')
    ax.set_ylabel('q')
    ax.set_zlabel('Weighted JS Distance')
    ax.set_title(f'Weighted JS Distance (alpha={alpha}) - New View')

    # 改变观察角度
    ax.view_init(elev=30, azim=45)  # 设置不同的俯仰和方位角

    plt.show()
