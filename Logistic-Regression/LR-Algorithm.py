# Created By Jacky on 2020/11/28

import numpy as np
import matplotlib.pyplot as plt
import random



def loadDataSet():
    dataMatrix = []
    dataLabel = []

    f = open('testSet.txt')

    for line in f.readlines():
        lineList = line.strip().split()

        # theta0  通常取得 1
        dataMatrix.append([1,float(lineList[0]),float(lineList[1])])
        dataLabel.append(int(lineList[2]))

    matLabel = np.mat(dataLabel).transpose()

    return dataMatrix , matLabel




# 定义 logistic-regression 的 sigmoid 函数
def sigmoid(inX):
    return 1 / (1 + np.exp(-inX))



# 采用梯度上升来求出最佳的 theta
def gradientAscent(dataMatrix,matLabel):
    m , n = np.shape(dataMatrix)
    matMatrix = np.mat(dataMatrix)

    # theta 出发点为 1
    w = np.ones((n,1))

    alpha = 0.001               # learning rate

    num = 500

    # 循环 500 次算出最佳的 theta
    for i in range(num):
        error = sigmoid(matMatrix * w) - matLabel
        w = w - alpha * matMatrix.transpose() * error

    return w




# Visualization
def plotter(weight):
    x0List = []
    y0List = []
    x1List = []
    y1List = []

    f = open('testSet.txt' , 'r')

    for line in f.readlines():
        lineList = line.strip().split()

        if lineList[2] == '0':
            x0List.append(float(lineList[0]))
            y0List.append(float(lineList[1]))
        else:
            x1List.append(float(lineList[0]))
            y1List.append(float(lineList[1]))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x0List, y0List, s = 10, c = 'red')
    ax.scatter(x1List, y1List, s = 10, c = 'green')

    xList = [];
    yList = []
    x = np.arange(-3, 3, 0.1)
    for i in np.arange(len(x)):
        xList.append(x[i])

    y = (-weight[0] - weight[1] * x) / weight[2]
    for j in np.arange(y.shape[1]):
        yList.append(y[0, j])

    ax.plot(xList, yList)
    plt.xlabel('x1');
    plt.ylabel('x2')
    plt.show()




# Test
if __name__ == '__main__':
    dataMatrix, matLabel = loadDataSet()
    weight = gradientAscent(dataMatrix,matLabel)

    print(weight)

    plotter(weight)