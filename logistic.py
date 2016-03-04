import pandas as pd
import numpy as np
import math
import time
import scipy.optimize
import scipy.special
from scipy.optimize import fmin_bfgs

alpha = 0.1
lamda = 0.001
e = 2.718281828459045235

x = pd.read_csv('subtrain.csv', usecols=['Round', 'FNL1', 'FNL2', 'FSP.1',
                                         'FSW.1', 'SSP.1', 'SSW.1', 'ACE.1', 'DBF.1', 'WNR.1', 'UFE.1', 'BPC.1', 'BPW.1', 'NPA.1', 'NPW.1', 'TPW.1',
                                         'ST1.1', 'ST2.1', 'ST3.1', 'ST4.1', 'ST5.1', 'FSP.2', 'FSW.2', 'SSP.2', 'SSW.2', 'ACE.2', 'DBF.2', 'WNR.2',
                                         'UFE.2', 'BPC.2', 'BPW.2', 'NPA.2', 'NPW.2', 'TPW.2', 'ST1.2', 'ST2.2', 'ST3.2', 'ST4.2', 'ST5.2'])

x2 = pd.read_csv('subtest.csv', usecols=['Round', 'FNL1', 'FNL2', 'FSP.1',
                                         'FSW.1', 'SSP.1', 'SSW.1', 'ACE.1', 'DBF.1', 'WNR.1', 'UFE.1', 'BPC.1', 'BPW.1', 'NPA.1', 'NPW.1', 'TPW.1',
                                         'ST1.1', 'ST2.1', 'ST3.1', 'ST4.1', 'ST5.1', 'FSP.2', 'FSW.2', 'SSP.2', 'SSW.2', 'ACE.2', 'DBF.2', 'WNR.2',
                                         'UFE.2', 'BPC.2', 'BPW.2', 'NPA.2', 'NPW.2', 'TPW.2', 'ST1.2', 'ST2.2', 'ST3.2', 'ST4.2', 'ST5.2'])

y = pd.read_csv('subtrain.csv', usecols=['Result'])
y2 = pd.read_csv('subtest.csv', usecols=['Result'])

# x = pd.read_csv('data/ex2data1.txt')
# x2 = pd.read_csv('data/ex2data2.txt')

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
    return 1.0 / (1 + np.exp(-z))


def hypothesis(X, theta):
    z = X.dot(theta[1:])
    return sigmoid(z)


def cost(theta):

    J = 0

    hyp = hypothesis(X, theta)

    j = 0
    for i in xrange(m):
        j += (math.log(hyp[i]) * Y[i]) + ((1 - Y[i]) * math.log(1 - hyp[i]))

    J = - (j / m)

    J = -J

    # print J

    return J


def gradient(theta):
    dx = np.zeros(X.shape[1], dtype=float)

    hyp = hypothesis(X, theta)

    # print (hyp-Y).shape,X.transpose().shape
    dx = dx + X.transpose().dot((hyp - Y))

    # dx = dx + X.dot(hyp - Y)
    dx = alpha * dx
    # print dx

    return dx


def descent_momentum(theta, v, mu):

    dx = gradient(theta)

    v[0] = mu[0] * v[0]
    v[1:] = (mu * v)[1:] - dx[:]
    theta += v

    # print theta,v,mu

    return theta, v, mu


def adagrad(theta):

    cache = np.zeros(x.shape[1])

    dx = gradient(theta)

    cache += dx**2

    theta += - dx / (np.sqrt(cache) + 1e-7)

    return theta


def logisitic():
    # print X
    v = np.zeros(X.shape[1]+1)
    mu = np.zeros(X.shape[1]+1)
    mu.fill(0.5)

    theta = np.zeros(X.shape[1]+1)
    # theta.fill(0)
    a = 0

    prev = cost(theta)
    # print "initial cost:", cost(theta)
    theta, v, mu = descent_momentum(theta, v, mu)

    while prev - cost(theta) >= 0.0000001:
        prev = cost(theta)
        theta, v, mu = descent_momentum(theta, v, mu)

        s = 0
        m = x2.shape[0]
        for i in xrange(m):

            xi = X2[i]
            hyp = hypothesis(xi, theta)
            yi = Y2[i]

            s += abs(absolute(hyp) - yi)

        print "epoch: (", a, ")",\
            "accuracy:", 1 - float(s) / float(m), "incorrect:", s[0]

        a += 1

    print theta
