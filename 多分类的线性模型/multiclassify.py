# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 16:35:52 2019
用于多分类的线性模型
应用一个三分类玩具数据集，二维特征数组，每个类别的数据均从不同的高斯分布中采样得到的
@author: Kylin
"""
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt 
import sys#将mglearn包的路径加入当前系统的路径中，否则默认在当前文件路径中
sys.path.insert(0, '../')
import mglearn
import numpy as np

#1. 加载数据集
X, y = make_blobs(random_state = 42)
#2. 可视化数据集
plt.rcParams['font.sans-serif']=['simhei'] #设置显示中文title的字体
fig = plt.figure("数据集可视化")
mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.title("数据集可视化")
plt.legend(["Class 0", "Class 1", "Class 2"])

#3. 在数据集上训练一个SVC分类器
linear_svm = LinearSVC().fit(X, y)
print("特征系数：")
print(linear_svm.coef_)
print("截距：")
print(linear_svm.intercept_)

#4. 三分类器的直线可视化
i = 0
linex = np.linspace(-15, 15)
for coef, intercept, color, style in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g'], ['-', '--', '-.']):
    liney = - (linex * coef[0] + intercept) / coef[1]
    label = "Line {}".format(i) 
    plt.plot(linex, liney, style, c=color, label = label)
    i += 1
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend()





