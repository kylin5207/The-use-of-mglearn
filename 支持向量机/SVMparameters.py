# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:46:57 2019
SVM调参
@author: Kylin
"""

import matplotlib.pyplot as plt
import sys#将mglearn包的路径加入当前系统的路径中，否则默认在当前文件路径中
sys.path.append("../")
import mglearn

fig, axes = plt.subplots(3, 3, figsize=(15,10))

#尝试三个不同的正则化系数
for ax, C in zip(axes, [-1, 0, 3]):
#尝试不同的gamma值，即用于控制高斯核的宽度
    for a, gamma in zip(ax, range(-1, 2)):
        mglearn.plots.plot_svm(log_C = C, log_gamma = gamma, ax=a)

axes[0,0].legend(["Class 0", "Class 1", "SV Class0", "SV Class1"], ncol=4, loc=(.9, 1.2))

