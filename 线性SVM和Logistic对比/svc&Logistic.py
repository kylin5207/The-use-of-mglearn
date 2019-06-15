# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 10:23:43 2019
最常见的两种线性分类算法Logistic回归与线性SVM
@author: Kylin
"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt 
import sys#将mglearn包的路径加入当前系统的路径中，否则默认在svcLogistic.py文件中
sys.path.insert(0, '../')
import mglearn

#获取数据集
X, y = mglearn.datasets.make_forge()

#展示两种方法对于决策边界的确定情况
fig, axes = plt.subplots(1, 2, figsize=(10,3))
for model, ax in zip([LogisticRegression(), LinearSVC()], axes):
    #默认情况下，都使用L2正则化
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax = ax, alpha=0.7)
    
    mglearn.discrete_scatter(X[:,0], X[:,1], y, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")

axes[0].legend()

#尝试调整正则化系数
plt.rcParams['font.sans-serif']=['simhei'] #设置显示中文title的字体
mglearn.plots.plot_linear_svc_regularization()
