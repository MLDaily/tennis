import pandas as pd
import numpy as np
import math
import time

alpha = 0.01
lamda = 0.001
e = 2.718281828459045235

x = pd.read_csv('data/ex2data1.txt')
x2 = pd.read_csv('data/ex2data2.txt')

X = x.values[:, :-1]
Y = x.values[:, -1]
X2 = x2.values[:, :-1]
Y2 = x2.values[:, -1]

m = X.shape[0]


def absolute(z):
    if z > 0.5:
        return 1
    return 0


def sigmoid(z):
    z = 1 / (1 + math.exp(-1.0 * z))
    return z


def hypothesis(xi, theta):

    z = 0
    for i in xrange(len(theta)):
        z += xi[i] * theta[i]

    return sigmoid(z)


def cost(theta):

    J = 0

    for i in range(m):

        yi = Y[i]
        xi = X[i]
        hyp = hypothesis(xi, theta)

        if yi == 1:
            a = math.log(hyp) * yi + lamda / (2 * m) * \
                np.sum(np.sum(x.values * x.values, axis=0), axis=0)
        else:
            a = (1 - yi) * math.log(1 - hyp) + lamda / (2 * m) * \
                np.sum(np.sum(x.values * x.values, axis=0), axis=0)

        J += (a / m)

    return J


def gradient(theta):
    l = np.zeros(X.shape[1], dtype=float)

    for i in range(m):
        yi = Y[i]
        xi = X[i]
        hyp = hypothesis(xi, theta)

        k = np.subtract(hyp, yi)
        l = np.add(l, np.multiply(k, xi))
        l = np.multiply(l, alpha)

    return l


def descent_momentum(theta, v, mu):

    dx = gradient(theta)

    v = mu * v - dx + lamda / (m) * \
        np.sum(np.sum(x.values, axis=0), axis=0)
    theta += v

    return theta, v, mu


def adagrad(theta):

    cache = np.zeros(x.shape[1])

    dx = gradient(theta)

    cache += dx**2

    theta += - dx / (np.sqrt(cache) + 1e-7) + lamda / (m) * \
        np.sum(np.sum(x.values, axis=0), axis=0)

    return theta


def main():
    v = np.zeros(X.shape[1])
    mu = np.zeros(X.shape[1])
    mu.fill(0.5)

    theta = np.random.randn(X.shape[1]) * 0.01
    a = 0
    
    prev = cost(theta)
    print "initial cost:", cost(theta)
    theta, v, mu = descent_momentum(theta, v, mu)

    while prev - cost(theta) <= 0.00001:
        prev = cost(theta)
        theta, v, mu = descent_momentum(theta, v, mu)

        s = 0
        m = x2.shape[0]
        for i in xrange(m):

            xi = X2[i]
            hyp = hypothesis(xi, theta)
            yi = Y2[i]

            s += abs(absolute(hyp) - yi)

        print "epoch: (", a, ") cost:", cost(theta)[0], \
            "accuracy:", 1 - float(s) / float(m), "incorrect:", s[0]
        
        a += 1

    print theta
