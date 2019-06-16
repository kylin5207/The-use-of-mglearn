# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 22:42:20 2019

@author: Kylin
"""
from sklearn.svm import SVC
import sys#将mglearn包的路径加入当前系统的路径中，否则默认在当前文件路径中
sys.path.insert(0, '../')
import mglearn
import matplotlib.pyplot as plt

#1.加载数据集
X, y = mglearn.tools.make_handcrafted_dataset()

#2.利用数据训练svm模型
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X,y)

mglearn.plots.plot_2d_separator(svm, X, eps=.5)
mglearn.discrete_scatter(X[:,0], X[:,1], y)

#3.画出支持向量
sv = svm.support_vectors_
#支持向量的类别标签由dual_coef_的正负号给出
sv_labels = svm.dual_coef_.ravel() > 0
mglearn.discrete_scatter(sv[:,0], sv[:,1], sv_labels, s=15, markeredgewidth=3)
plt.title("Support Vectors")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

