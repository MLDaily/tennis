import pandas as pd
import numpy as np
import math
import time

alpha = 0.5
e = 2.718281828459045235

x = pd.read_csv('Data/AusOpen-men-2013.csv',usecols=['Round','FNL1','FNL2','FSP.1',\
	'FSW.1','SSP.1','SSW.1','ACE.1','DBF.1','WNR.1','UFE.1','BPC.1','BPW.1','NPA.1','NPW.1','TPW.1',\
	'ST1.1','ST2.1','ST3.1','ST4.1','ST5.1','FSP.2','FSW.2','SSP.2','SSW.2','ACE.2','DBF.2','WNR.2',\
	'UFE.2','BPC.2','BPW.2','NPA.2','NPW.2','TPW.2','ST1.2','ST2.2','ST3.2','ST4.2','ST5.2'])

y = pd.read_csv('Data/AusOpen-men-2013.csv',usecols=['Result'])



m = x.shape[0]

def sigmoid(z):
	# for v in range(len(z)):
	# 	z[v] = 1/(1+np.power(e,-z[v]))

	z = 1/(1+np.power(e,-z))

	return z

def hypothesis(x,i,theta):

	xi = x[i:i+1].values[0]
	tran = np.transpose(theta)

	# print theta, tran, xi, np.dot(tran,xi)
	# time.sleep(2)

	z = np.dot(tran,xi)
	# print z
	# time.sleep(2)
	return sigmoid(z)

def cost_function(theta):

	J = 0

	for i in range(m):

		yi = y[i:i+1].values[0]
		xi = x[i:i+1].values[0]
		hyp = hypothesis(x,i,theta)
		# print hyp

		a = np.multiply( np.log10(hyp), yi)
		b = np.multiply( (1-yi), np.log10( np.subtract( 1, hyp ) ) )

		# print a
		# time.sleep(2)

		c = np.add(a,b)
		J += np.divide(c,m)

	return J

def descent(theta):
	
	l = np.zeros(x.shape[1])

	for i in range(m):

		yi = y[i:i+1].values[0]
		xi = x[i:i+1].values[0]
		hyp = hypothesis(x,i,theta)
		k = np.subtract( hyp, yi )
		l = np.add(l, np.multiply( k, xi ))

	l = np.multiply(alpha,l)

	theta = np.subtract(theta,l) 

	return theta

if __name__ == '__main__':
	theta = np.ones(x.shape[1])
	print descent(theta)
	print cost_function(theta)