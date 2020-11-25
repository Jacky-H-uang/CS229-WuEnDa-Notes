# Created By Jacky on 2020/11/25


import numpy as np
import matplotlib.pyplot as plt
import random
import math

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    print(numFeat)
    dataMat = []; labelMat = []

    fr = open(fileName)

    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat




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
def lwrTest(textArr,xArr,yArr,k):
    # 测试点的个数
    m = np.shape(textArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwrFunction(textArr[i], xArr, yArr, k)

    return yHat


# Visualization
def plotter():
    dataSet , Labels = loadDataSet('test0.txt')

    plt.scatter([x[1] for x in dataSet],Labels)

    # 返回x排序后的下标
    xIndex = np.array([x[1] for x in dataSet]).argsort()
    xSort = np.array([x[1] for x in dataSet])[xIndex]
    ySort = lwrTest(dataSet, dataSet, Labels, 0.01)[xIndex]
    plt.plot(xSort, ySort, 'r')
    plt.show()


if __name__ == '__main__':
    plotter()