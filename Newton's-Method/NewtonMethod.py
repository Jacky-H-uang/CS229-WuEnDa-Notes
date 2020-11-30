# Created By Jacky on 2020/11/30


"""
牛顿迭代法是在找到一个 theta 使得 f(theta) = 0 (即函数的零点)

所以每次迭代的时候都使得一阶导数最大 maxmize{l(theta)}

相关公式：
    diff = Θ1 - Θ0

    f(Θ0)' = f(Θ0) / diff

    diff = f(Θ0) / f(Θ0)'

    所以关于 Θ 的迭代公式：
        Θ(t+1) = Θ(t) - f(Θ(t)) / f(Θ(t))'
"""

import numpy as np
import matplotlib.pyplot as plt



# Newton's Method
def newton_method(theta, X, y, num_iter):
    loss_history = np.zeros(num_iter)
    theta_history = np.zeros((num_iter, 2))
    m = len(y)
    for i in range(num_iter):
        y_pred = np.dot(theta, X)
        theta = theta - (np.dot(theta,X)-y).dot(X.T).dot(np.linalg.inv(np.dot(X,X.T)))
        loss = 1/(2*m) * np.sum(np.square(y_pred - y))
        theta_history[i, :] = theta
        loss_history[i] = loss
        if i%5 == 1:
            print('theta is',theta)
            print('current loss is',loss)
    return theta, theta_history, loss_history





if __name__ == '__main__':
    X = np.array([np.ones(100), np.random.rand(100)])
    y = np.dot([4, 3], X) + np.random.rand(100)
    print('x size is', X.size)
    plt.scatter(X[1, :], y)
    plt.show()

    print(X.shape)
    print(y.shape)
    print('newton_method start')
    num_iter = 20


    theta_ini = np.array([np.random.randn(2)])
    theta, theta_history, loss_history = newton_method(theta_ini,X,y, num_iter)

    residual = np.zeros((len(y)-1))

    print(theta)

    plt.plot(loss_history)
    plt.show()