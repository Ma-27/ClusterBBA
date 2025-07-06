# -*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline  # spline

from utility.plot_style import apply_style

apply_style()


def BJS(a, b):  # 计算BPA分布情况 基于BJS
    Sum = 0
    for i in range(len(a)):
        Sum += (a[i] * np.log10((2 * a[i]) / (a[i] + b[i] + 10 ** -12) + 10 ** -12) + (
                b[i] * np.log10((b[i] * 2) / (a[i] + b[i] + 10 ** -12) + 10 ** -12)))
    Sum *= 0.5
    return Sum


def BRE(a, b):  # 计算BPA分布情况 基于BRE
    Sum = 0
    for i in range(len(a)):
        Sum += a[i] * np.log10(a[i] / (np.sqrt(a[i] * b[i]) + 10 ** -12) + 10 ** -12)
    Sum1 = 0
    for i in range(len(a)):
        Sum1 += b[i] * np.log10(b[i] / (np.sqrt(a[i] * b[i]) + 10 ** -12) + 10 ** -12)
    Sum *= Sum1
    Sum = np.sqrt(Sum)
    return Sum


def DHM1(a, b):  # 计算BPA分布情况 基于DHM
    Sum = 0
    Sum0 = 0
    for i in range(len(a)):
        Sum += a[i] * np.log((a[i] + b[i]) / (2 * b[i] + 1e-12) + 1e-12)
        Sum0 += b[i] * np.log((a[i] + b[i]) / (2 * a[i] + 1e-12) + 1e-12)
    return 2 / (1e-12 + 1 / (Sum + 1e-12) + 1 / (Sum0 + 1e-12))


def DCM(a, b):  # 计算BPA分布情况 基于DHM
    Sum = 0
    Sum0 = 0
    for i in range(len(a)):
        Sum += a[i] * np.log(1e-12 + a[i] * (a[i] + b[i]) / (a[i] ** 2 + b[i] ** 2 + 1e-12))
        Sum0 += b[i] * np.log(1e-12 + b[i] * (a[i] + b[i]) / (a[i] ** 2 + b[i] ** 2 + 1e-12))
    return (Sum ** 2 + Sum0 ** 2) / (Sum + Sum0)  ##Sum+Sum0-2*Sum*Sum0/(Sum+Sum0)


def DGM(a, b):  # 计算BPA分布情况 基于DHM
    Sum = 0
    Sum0 = 0
    for i in range(len(a)):
        Sum += a[i] * np.log(1e-12 + np.sqrt(2 * a[i] ** 2 / (a[i] ** 2 + b[i] ** 2 + 1e-12)))
        Sum0 += b[i] * np.log(1e-12 + np.sqrt(2 * b[i] ** 2 / (a[i] ** 2 + b[i] ** 2 + 1e-12)))
    return np.sqrt((Sum ** 2 + Sum0 ** 2) / 2)


def Cos(a, b):
    Multi_value = 0
    Add1_value = 0
    Add2_value = 0
    for i in range(len(a)):
        Multi_value += a[i] * b[i]
        Add1_value += a[i] * a[i]
        Add2_value += b[i] * b[i]
    Cos_value = Multi_value / (np.sqrt(Add1_value) * np.sqrt(Add2_value))
    return 1.0 - Cos_value


def Dis(a, b):
    Sum = []
    for i in range(len(a)):
        Sum.append(a[i] - b[i])
    Sum1 = 0
    for i in range(len(a)):
        Sum1 += Sum[i] * Sum[i]
    return np.sqrt(Sum1 / 2.0)


result0 = []
result = []
result1 = []
result2 = []
ax_x = []
ax_x1 = []
x = [0.5, 0.5]
for i in range(101):
    y0 = i / 100.0
    y1 = (100.0 - i) / 100.0
    y = [y0, y1]
    result0.append(Cos(x, y))
for i in range(101):
    y0 = i / 100.0
    y1 = (100.0 - i) / 100.0
    y = [y0, y1]
    result.append(Dis(x, y))
    ax_x.append(y0)
for i in range(101):
    y0 = i / 100.0
    y1 = (100.0 - i) / 100.0
    y = [y0, y1]
    result1.append(BRE(x, y))
    ax_x1.append(y0)
for i in range(101):
    y0 = i / 100.0
    y1 = (100.0 - i) / 100.0
    y = [y0, y1]
    result2.append(BJS(x, y))
ax_x = np.array(ax_x)
ax_x1 = np.array(ax_x1)
result = np.array(result)
x_new = np.linspace(ax_x.min(), ax_x.max(), 300)
power_smooth = make_interp_spline(ax_x, result)(x_new)
result1 = np.array(result1)
x_new1 = np.linspace(ax_x1.min(), ax_x1.max(), 300)
power_smooth1 = make_interp_spline(ax_x1, result1)(x_new1)
plt.Figure()
# plt.plot(x_new, power_smooth,color='black')
# plt.plot(x_new, power_smooth,color='black')
plt.plot(ax_x, result, color='red', label='Dis')
plt.plot(ax_x, result0, color='black', label='Cos')
plt.plot(ax_x, result2, color='green', label='BJS')
plt.plot(ax_x, result1, color='blue', label='BRE')

# plt.plot(ax_x1, result2,color='r')
plt.title("Different similarity measurement with changing "r'$\alpha$')
plt.legend()
plt.axis([0.0, 1.0, 0.0, 1.0])
plt.xlabel(r'$\alpha$')
plt.ylabel('similarity Value')
plt.savefig("../results/dsmchange2.png")
