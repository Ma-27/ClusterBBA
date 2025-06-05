# -*- coding: UTF-8 -*-
import numpy as np


def allSubsets(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]


# Import data description csv:
# the first row and the first column: custom name; the first row and the second column: a set of hypothetical events represented by a single letter, and the connection is represented as a string
# Each column in the second row represents the events corresponding to all the estimated values of the bpa function m, and the probability of the event corresponding to the corresponding bpa is listed in the lower row.

Load_dir = "../data/muti sensor target recognition.csv"
raw = np.loadtxt(open(Load_dir, "rb"), dtype=str, delimiter=",", skiprows=0)
Events = set()
Events.update(raw[0, 1])
Events_list = list(Events)
All_Subsets = list(allSubsets(Events))
Bpa_for_elements = list(raw[1, :])
All_BPA = raw[2:, :].astype(np.float)
M_Sum = All_BPA.shape[0]
Bpa_Array = np.zeros((M_Sum, 2 ** len(Events_list)))
for i in range(M_Sum):
    for j in range(All_BPA.shape[1]):
        cur_set = set()
        cur_set.update(Bpa_for_elements[j])
        for index, value in enumerate(All_Subsets):
            compare_set = set()
            compare_set.update(set(value))
            if cur_set.issubset(compare_set) and compare_set.issubset(cur_set):
                Bpa_Array[i, index] = All_BPA[i, j]

Base_m = Bpa_Array[0, :]
print(All_Subsets)
for m in range(1, M_Sum):
    K = 0
    print("D-S times:", m)
    other_m = Bpa_Array[m, :]
    for i, b_value in enumerate(All_Subsets):
        cur_set = set()
        cur_set.update(b_value)
        for j, c_value in enumerate(All_Subsets):
            compare_set = set()
            compare_set.update(c_value)
            if len(cur_set & compare_set) == 0:
                K += Base_m[i] * other_m[j]
    _k = 1 / (1 - K)
    temp_m = np.zeros_like(Base_m)
    for i, a_value in enumerate(All_Subsets):
        a_set = set()
        a_set.update(a_value)
        if len(a_set) == 0:
            continue
        for j, b_value in enumerate(All_Subsets):
            b_set = set()
            b_set.update(b_value)
            for p, c_value in enumerate(All_Subsets):
                c_set = set()
                c_set.update(c_value)
                b_c = b_set & c_set
                if a_set.issubset(b_c) and b_c.issubset(a_set):
                    temp_m[i] += Base_m[j] * other_m[p]
        temp_m[i] *= _k
    Base_m = temp_m
    print("current combined bpa:", temp_m)
