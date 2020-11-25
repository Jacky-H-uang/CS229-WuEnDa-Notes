# Created By Jacky on 2020/11/20

import numpy as np
import matplotlib.pyplot as plt
import random


"""
简单的拟合 
y = theta0 + theta1 * x(i)
"""


# 拟合的点 40 个
m = 40

# 定义代价函数
def costFunction(X,Y,theta):
    diff = np.dot(X,theta) - Y

    return (1/(2*m)) * np.dot(np.transpose(diff),diff)




# 对 cost-function 求偏导数
def derivativeFunction(X,Y,theta):
    diff = np.dot(X,theta) - Y

    # X^T * (X(Θ) - 向量Y)
    return (1/m) * np.dot(np.transpose(X),diff)





# 梯度下降的迭代
def gradientFunction(X,Y,alpha):
    # 先定义一个开始的梯度
    # [
    #   [1],
    #   [1]
    # ]
    theta = np.array([1,1]).reshape(2,1)
    gradient = derivativeFunction(X,Y,theta)

    # 假设每一次的梯度迭代中 , 如果全部都小于 1e-5 则求得局部最优解
    while not all(abs(gradient) <= 1e-5):
        theta = theta - alpha * gradient
        gradient = derivativeFunction(X,Y,theta)

    return theta




# Viusalization
def plotter(X,Y,theta):
    ax = plt.subplot(111)
    ax.scatter(X,Y,s = 30,c = 'red')
    plt.xlabel("X")
    plt.ylabel("Y")



    # 拟合这条曲线
    x = np.arange(0,21,0.1)
    y = theta[0] + theta[1] * x
    ax.plot(x,y)
    plt.show()



# 产生拟合的曲线 Demo
def createDemo():
    real_func = np.poly1d([5, 10])

    x_points = list()
    for i in range(m):
        x_points.append(random.uniform(0, 21))
    y_points = list()
    for i in range(m):
        y_points.append(real_func(x_points[i]))

    # 加上噪点
    y_Noise = [np.random.normal(0,1) + y for y in y_points]

    # X0 规定初始化为 1
    X0 = np.ones((m, 1))                    # 生成 m 行 1 列的向量全为 1
    X1 = np.array(x_points).reshape(m, 1)   # 生成 m 行 1 列的向量

    # 按照堆叠形成数据样本
    # 把 X0 和 X1 的列水平排列起来 行不变
    # 这里把 X0 和 X1 堆叠成数据 m 行 2 列
    X = np.hstack((X0, X1))

    # 对应于的 Y 的坐标
    Y = np.array(y_Noise).reshape(m,1)

    return X1,X,Y





# Main
if __name__ == '__main__':
    # learning rate
    alpha = 0.01

    X1,X,Y = createDemo()

    opt = gradientFunction(X,Y,alpha)
    plotter(X1,Y,opt)