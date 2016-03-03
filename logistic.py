import pandas as pd
import numpy as np
import math
import time

alpha = 1
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

m = x.shape[0]


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

        yi = y[i:i + 1].values[0]
        xi = x[i:i + 1].values[0]
        hyp = hypothesis(xi, theta)

        if yi == 1:
            a = math.log(hyp) * yi
        else:
            a = (1 - yi) * math.log(1 - hyp)

        J += (a / m)

    return J


def descent_momentum(theta, v, mu):

    l = np.zeros(x.shape[1], dtype=float)

    for i in range(m):
        yi = y[i:i + 1].values[0]
        xi = x[i:i + 1].values[0]
        hyp = hypothesis(xi, theta)

        k = np.subtract(hyp, yi)
        l = np.add(l, np.multiply(k, xi))
        l = np.multiply(l, alpha)

    v = mu * v - l
    theta += v

    return theta, v, mu


def adagrad(theta):
    l = np.zeros(x.shape[1], dtype=float)

    cache = np.zeros(x.shape[1])

    for i in range(m):
        yi = y[i:i + 1].values[0]
        xi = x[i:i + 1].values[0]
        hyp = hypothesis(xi, theta)

        k = np.subtract(hyp, yi)
        l = np.add(l, np.multiply(k, xi))
        cache += l**2
        l = np.multiply(l, alpha)

    theta += - l / (np.sqrt(cache) + 1e-7)

    return theta


def main():
    theta = np.random.randn(x.shape[1]) * 0.01
    a = 0
    prev = cost(theta)
    print "epoch: initial cost:", cost(theta)[0], "theta mean:", np.mean(theta)
    theta = adagrad(theta)

    while prev != cost(theta):
        prev = cost(theta)
        theta = adagrad(theta)

        s = 0
        m = x2.shape[0]
        for i in xrange(m):

            xi = x2[i:i + 1].values[0]
            hyp = hypothesis(xi, theta)
            yi = y2[i:i + 1].values[0]

            s += abs(absolute(hyp) - yi)
        print s

        print "epoch: (", a, ") cost:", cost(theta)[0], "theta STD:", np.std(theta),\
            "accuracy:", 1 - float(s) / float(m)
        if a > 15 and prev > cost(theta):
            break
        a += 1

    print theta


# Output
# epoch: ( 22 ) cost: -1.83812306886 theta mean: 0.000571528008675 accuracy: 0.431420543195
# epoch: ( 16 ) cost: -1.84022982028 theta STD: 0.0441482809193 accuracy:
# 0.448643985393
# epoch: ( 11 ) cost: -1.42125360658 theta STD: 0.032630121632 accuracy: 0.585106382979
# epoch: ( 431 ) cost: -1.15794385906 theta STD: 0.0201382753123 accuracy:
# 0.542553191489
# epoch: ( 46 ) cost: -0.842040788256 theta STD: 0.0101694117625 accuracy: 0.510638297872
# epoch: ( 15 ) cost: -0.833312496603 theta STD: 0.00999654906749 accuracy: 0.531914893617
