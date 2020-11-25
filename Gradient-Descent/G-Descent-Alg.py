# Created By Jacky on 2020/11/16

"""
This is Gradient-Descent Algorithm's Implemention
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt



# 定义一个 20 个点的数据集
m = 20

# X0 规定初始化为 1
X0 = np.ones((m,1))                     # 生成 m 行 1 列的向量全为 1
X1 = np.arange(1,m+1).reshape(m,1)      # 生成 m 行 1 列的向量从 1 到 m


# 按照堆叠形成数据样本
# 把 X0 和 X1 的列水平排列起来 行不变
# 这里把 X0 和 X1 堆叠成数据 m 行 2 列
X = np.hstack((X0,X1))


# 对应于的 Y 的坐标
Y = np.array(
        [3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,11, 13, 13, 16, 17, 18, 17, 19, 21]
    ).reshape(m,1)



# learning rate
alpha = 0.01



"""
Define the cost-function
"""
def costFunction(theta,X,Y):
    diff = np.dot(X,theta) - Y
    # 数组需要像矩阵那样相乘的时候就需要用到 dot()
    return (1/(2*m)) * np.dot(np.transpose(diff),diff)




"""
Define the grandient-function for cost-function
"""
def gradientFunction(theta , X, Y):
    diff = np.dot(X,theta) - Y

    return (1/m) * np.dot(np.transpose(X) , diff)



"""
Gradient-Descent Iterator
"""
def gradientDescent(X , Y , alpha):
    # theta 初始化为 [
    #                 [1],
    #                 [1]
    #               ]
    theta = np.array([1,1]).reshape(2,1)
    gradient = gradientFunction(theta,X,Y)

    # 假设处于小于 1e-5 次方的时候就找到局部的最优解
    # 没找到局部的最优解即需要继续迭代
    while not all(abs(gradient) <= 1e-5):
        theta = theta - alpha * gradient
        gradient = gradientFunction(theta , X , Y)

    return theta




"""
Visualization
"""
def plotter(X,Y,theta):
    ax = plt.subplot(111)
    ax.scatter(X,Y,s = 30,c = 'red',marker = "s")
    plt.xlabel("X")
    plt.ylabel("Y")
    # 拟合这条曲线
    x = np.arange(0,21,0.2)
    y = theta[0] + theta[1]*x
    ax.plot(x,y)
    plt.show()


if __name__ == '__main__':
    optimal = gradientDescent(X, Y, alpha)
    plotter(X1,Y,optimal)