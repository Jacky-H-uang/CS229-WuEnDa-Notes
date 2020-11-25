# Created By Jacky on 2020/11/23


"""
1. Cost Function :
        Fit the Θ to minimize
        J(Θ) = W(i)∑ i=1~m (y(i) - ΘT*x(i))^2
        其中的 W(i) 表示权值

2. 关于权重的函数选择
        权重一般选择符合高斯分布
        W(i) = e ^[-(X(i)-X)^2 / 2t^2]          (weight function)
        参数 t 控制权值变化的速率

3. 大致推导过程：分布式高斯分布 --> "likehood" --> "log likehood" --> "MLE" --> 推导出 Cost Function

4. 写成矩阵的表达式：
    w = (X^T WX)^-1 X^T Wy
"""


import numpy as np
import matplotlib.pyplot as plt
import random


def real_func(x):
    return np.exp(-2*np.sin(x))


# 产生拟合的曲线 Demo 500 个点
def createDemo():
    m = 500

    x_points = list()
    for i in range(m):
        x_points.append(random.uniform(-10,10))
    y_points = list()
    for i in range(m):
        y_points.append(float(real_func(x_points[i])))

    # 加上噪点
    y_Noise = [float(np.random.normal(0,0.1)) + y for y in y_points]

    # X0 规定初始化为 1
    X0 = [1] * m
    X1 = x_points
    X = list()
    for i in range(m):
        aux = []
        aux.append(X0[i])
        aux.append(X1[i])
        X.append(aux)


    return X1,X,y_Noise




# 利用高斯分布的权重函数来算出拟合点
def lwrFunction(testPoint,xArr,yArr,k):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)

    # 数据点的个数
    m = np.shape(xMat)[0]

    # 初始化一个阶数等于m的方阵，其对角线的值为1，其余值均为0
    weights = np.mat(np.eye(m))

    # theta = (X^T * W * X) / (X^T * W * Y)
    # 根据权重来更新每一个点
    for j in range(m):
        diffMat = testPoint - xMat[j,:]

        # 更新权重 满足高斯分布
        weights[j,j] = np.exp(diffMat * diffMat.T / (-2.0 * k**2))

    xTx = xMat.T * (weights * xMat)

    # print("weights",weights)
    # 通过计算行列式的值来判是否可逆 (分母不能为 0)
    if np.linalg.det(xTx) == 0.0:
        print("this matrix is singular,cannot do inverse")
        return
    ws =xTx.I * (xMat.T * (weights * yMat.T))

    return testPoint * ws




# 返回拟合的 y
def lwrDemo(textArr,xArr,yArr,k):
    # 测试点的个数
    m = np.shape(textArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwrFunction(textArr[i], xArr, yArr, k)

    return yHat




# Visualization
def plotter():
    X1 , X , Y = createDemo()
    plt.scatter(X1,Y)

    xIndex = np.array([x[1] for x in X]).argsort()
    xSort = np.array([x[1] for x in X])[xIndex]
    ySort = lwrDemo(X,X,Y,0.1)[xIndex]
    plt.plot(xSort,ySort,'r')
    plt.show()






if __name__ == '__main__':
    plotter()