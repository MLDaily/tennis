import pandas as pd
import numpy as np
import math
import time
import scipy.optimize
import scipy.special
from scipy.optimize import fmin_bfgs
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

alpha = 10
lamda = 0.001
e = 2.718281828459045235

# df = pd.read_csv('subtrain.csv', usecols=['FNL1', 'FNL2','Result'])


# x = pd.read_csv('subtrain.csv', usecols=['FNL1', 'FNL2'])

# x2 = pd.read_csv('subtest.csv', usecols=['Round', 'FNL1', 'FNL2', 'FSP.1',
#                                          'FSW.1', 'SSP.1', 'SSW.1', 'ACE.1', 'DBF.1', 'WNR.1', 'UFE.1', 'BPC.1', 'BPW.1', 'NPA.1', 'NPW.1', 'TPW.1',
#                                          'ST1.1', 'ST2.1', 'ST3.1', 'ST4.1', 'ST5.1', 'FSP.2', 'FSW.2', 'SSP.2', 'SSW.2', 'ACE.2', 'DBF.2', 'WNR.2',
#                                          'UFE.2', 'BPC.2', 'BPW.2', 'NPA.2', 'NPW.2', 'TPW.2', 'ST1.2', 'ST2.2', 'ST3.2', 'ST4.2', 'ST5.2'])

# y = pd.read_csv('subtrain.csv', usecols=['Result'])
# y2 = pd.read_csv('subtest.csv', usecols=['Result'])

df = pd.read_csv('data/ex2data2.txt')
x = pd.read_csv('data/ex2data2.txt')
# x2 = pd.read_csv('data/ex2data1.txt')

X = x.values[:,:-1]
Y = x.values[:,-1]

m = X.shape[0]
n = X.shape[1]


def absolute(z):
    # sfor i in z:
    if z > 0.5:
        z = 1
    else:
        z = 0

    return z


def sigmoid(z):
    z = 1.0 / (1 + np.exp(-z))
    # print z
    return z


def hypothesis(X, theta):
    z = X.dot(theta[1:])
    return sigmoid(z)


def hyp_linear(X, theta):
    # print theta
    z = np.zeros((X.shape[0]))
    for i in xrange(X.shape[0]):
        z[i] = theta[0] + X[i, :].dot(theta[1:])

    # print z

    return z


def cost_linear(theta):
    J = 0

    hyp = hyp_linear(X, theta)

    j = 0
    for i in xrange(m):
        j += (hyp[i] - Y[i])**2

    J = j / (2 * m)

    return J


def cost(theta):

    J = 0

    hyp = hypothesis(X, theta)

    j = 0
    for i in xrange(m):
        # print j,hyp[i],Y[i]
        if hyp[i] == 1:
            j += - math.log(hyp[i]) * Y[i]
        else:
            j += - (1 - Y[i]) * math.log(1 - hyp[i])

    J = j / m

    return J


def gradient(theta):
    dx = np.zeros((n + 1), dtype=float)

    hyp = hypothesis(X, theta)

    print hyp[0], Y[0]
    dx[0] = dx[0] + ((hyp[0] - Y[0]))
    ddx = dx[1:]
    for i in xrange(0, n):
        for j in xrange(m):
            ddx[i] = ddx[i] + X[j][i] * ((hyp[i] - Y[i]))
        ddx[i] = ddx[i] / m

    dx[1:] = ddx
    return dx


def descent(theta):
    dx = gradient(theta)
    # theta[0] = theta[0] + alpha
    theta += alpha * dx
    print theta
    return theta


def descent_momentum(theta, v, mu):

    dx = gradient(theta)

    # v[0] = mu[0] * v[0]
    v = (mu * v) - dx
    theta += v

    # print v

    return theta, v, mu


def adagrad(theta):

    cache = np.zeros(n + 1)

    dx = gradient(theta)

    cache += dx**2

    theta += - dx / (np.sqrt(cache) + 1e-7)

    return theta


def logisitic():

    v = np.zeros(n + 1)
    mu = np.zeros(n + 1)
    mu.fill(0.5)

    theta = np.zeros(n + 1)
    a = 0

    prev = cost(theta)
    print "initial cost:", prev
    theta = descent(theta)

    while 1:
        prev = cost(theta)
        theta = descent(theta)

        # accuracy = 1 - float(s) / float(m1)
        print "epoch: (", a, ")", "cost:", cost(theta)

        if abs(prev - cost(theta)) <= 0.00001:
            break

        a += 1

    # print theta
    visual(X, theta)


def visual(X, theta):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # df.plot(kind='bar')

    ax = df.loc[df["Output"] == 0].plot(
        kind='scatter', x=0, y=1, color='DarkGreen', label='Group 1')
    df.loc[df["Output"] == 1].plot(
        kind='scatter', x=0, y=1,marker='+', color='DarkBlue', label='Group 2', ax=ax)

    y = np.zeros((X.shape[0]))
    a = theta[1] / theta[2]
    b = theta[0] / theta[2]
    for i in xrange(X.shape[0]):
        y[i] = - a * X[i, 0] - b + 120
    ax.plot(y, X[:, 0])

    plt.savefig("random")
    plt.close()
