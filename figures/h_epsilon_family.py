# -*- coding: utf-8 -*-
"""Visualise the family of $h_\varepsilon(m_i(A))$ curves.

This is an updated version of the original plotting script with a
denser sweep over ``delta`` and ``epsilon``.  It follows the example
provided in the reviewer comment to better illustrate the smoothing
behaviour of the function

.. math::

    h(m) = \frac{1}{1 + \\exp\bigl(-(m - \\delta)/\varepsilon\bigr)}.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import SCALE_DELTA, SCALE_EPSILON
from utility.plot_style import apply_style
from utility.plot_utils import savefig

apply_style()


def h_epsilon(x: np.ndarray, delta: float, epsilon: float) -> np.ndarray:
    """Compute the smoothed count for mass ``x``."""
    return 1.0 / (1.0 + np.exp(-(x - delta) / epsilon))


def plot_family(out_path: str) -> None:
    base_delta = SCALE_DELTA
    base_epsilon = SCALE_EPSILON
    x = np.linspace(0.0, 0.02, 500)

    fig, ax = plt.subplots(figsize=(8, 6))

    for d in np.linspace(6e-3, 10e-3, 25):
        ax.plot(x, h_epsilon(x, d, base_epsilon), linewidth=1)

    for e in np.linspace(2e-3, 6e-3, 25):
        ax.plot(x, h_epsilon(x, base_delta, e), linewidth=1)

    ax.set_xlabel(r"$m_i(A)$")
    ax.set_ylabel(r"$h_{\varepsilon}(m_i(A))$")
    ax.set_xlim(0.0, 0.02)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    ax.set_title(r"Dense Family of $h_{\varepsilon}$ Curves for Varying $\delta$ and $\varepsilon$")

    savefig(fig, out_path)


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(base, "..", "experiments_result")
    os.makedirs(res_dir, exist_ok=True)
    plot_family(os.path.join(res_dir, "h_epsilon_family.png"))
