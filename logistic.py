import pandas as pd
import numpy as np
import math
import time
from decimal import *

alpha = 0.01
e = 2.718281828459045235

x = pd.read_csv('subtrain.csv',usecols=['Round','FNL1','FNL2','FSP.1',\
	'FSW.1','SSP.1','SSW.1','ACE.1','DBF.1','WNR.1','UFE.1','BPC.1','BPW.1','NPA.1','NPW.1','TPW.1',\
	'ST1.1','ST2.1','ST3.1','ST4.1','ST5.1','FSP.2','FSW.2','SSP.2','SSW.2','ACE.2','DBF.2','WNR.2',\
	'UFE.2','BPC.2','BPW.2','NPA.2','NPW.2','TPW.2','ST1.2','ST2.2','ST3.2','ST4.2','ST5.2'])

x2 = pd.read_csv('subtest.csv',usecols=['Round','FNL1','FNL2','FSP.1',\
	'FSW.1','SSP.1','SSW.1','ACE.1','DBF.1','WNR.1','UFE.1','BPC.1','BPW.1','NPA.1','NPW.1','TPW.1',\
	'ST1.1','ST2.1','ST3.1','ST4.1','ST5.1','FSP.2','FSW.2','SSP.2','SSW.2','ACE.2','DBF.2','WNR.2',\
	'UFE.2','BPC.2','BPW.2','NPA.2','NPW.2','TPW.2','ST1.2','ST2.2','ST3.2','ST4.2','ST5.2'])

y = pd.read_csv('subtrain.csv',usecols=['Result'])
y2 = pd.read_csv('subtest.csv',usecols=['Result'])

m = x.shape[0]

def sigmoid(z):
	# for v in range(len(z)):
	# 	z[v] = 1/(1+np.power(e,-z[v]))

	z = 1/(1+np.power(e,-z))

	return z

def hypothesis(x,i,theta):

	xi = x[i:i+1].values[0]
	tran = np.transpose(theta)

	# print xi, i
	# time.sleep(2)

	z = np.dot(tran,xi)
	# print z
	# time.sleep(2)
	return sigmoid(z)

def cost(theta):

	J = 0

	for i in range(m):

		yi = y[i:i+1].values[0]
		xi = x[i:i+1].values[0]
		hyp = hypothesis(x,i,theta)
		# print np.log(1-hyp) * (1-yi), yi, np.log( hyp ) * yi

		if yi != 0:
			a = np.multiply( np.log( hyp ), yi)
		else:
			a = np.multiply( (1-yi), np.log( 1-hyp ) )

		J += np.divide(a,m)
		# print J

	return J

def descent(theta):
	
	l = np.zeros(x.shape[1],dtype=float)

	for i in range(m):
		yi = y[i:i+1].values[0]
		xi = x[i:i+1].values[0]
		hyp = hypothesis(x,i,theta)

		k = np.subtract( hyp, yi )
		l = np.add(l, np.multiply( k, xi ))
		l = np.multiply(alpha,l)
		# print hyp,k,l
	# time.sleep(10)
	theta = np.subtract(theta,l) 

	return theta

if __name__ == '__main__':
	theta = np.ones(x.shape[1])
	for i in range(100):
		prev = cost(theta)
		while prev < cost(theta):
			theta = descent(theta)
		print cost(theta)
	s = 0
	m = x2.shape[0]
	for i in range(m):
		hyp = hypothesis(x,i,theta)
		yi = y2[i:i+1].values[0]
		# print int(hyp), yi,i
		s += abs(int(hyp) - yi)
	print float(s)/float(m), m