# -*- coding: utf-8 -*-
"""DS+SVM 证据化基线包的初始化。"""

from .application_dssvm import evaluate_on_dataset

__all__ = [
    "evaluate_on_dataset",
    "application_dssvm",
    "trainer",
    "calibration_logistic",
    "calibration_multinomial",
    # ……其他你想公开的模块名
]
