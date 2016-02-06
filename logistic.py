import pandas as pd
import numpy as np
import math
import time

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

def absolute(z):
	if z > 0.5:
		return 1
	return 0

def sigmoid(z):
	z = 1/(1+math.exp(-1.0*z))
	return z

def hypothesis(xi,theta):

	z = 0
	for i in xrange(len(theta)):
		z += xi[i] * theta[i]
		# print xi[i], theta[i], z
	
	return sigmoid(z)

def cost(theta):

	J = 0

	for i in range(m):

		yi = y[i:i+1].values[0]
		xi = x[i:i+1].values[0]
		hyp = hypothesis(xi,theta)

		if yi == 1:
			a =  math.log( hyp ) * yi
		else:
			a =  (1-yi) * math.log( 1-hyp )

		J +=  (a / m) 

	return J

def descent(theta):
	
	l = np.zeros(x.shape[1],dtype=float)

	for i in range(m):
		yi = y[i:i+1].values[0]
		xi = x[i:i+1].values[0]
		hyp = hypothesis(xi,theta)

		k = np.subtract( hyp, yi )
		l = np.add(l, np.multiply( k, xi ))
		l = np.multiply(alpha,l)
	
	theta = np.subtract(theta,l)

	return theta

if __name__ == '__main__':
	theta = np.zeros(x.shape[1])
	prev = cost(theta)
	print prev
	theta = descent(theta)
	# print theta

	i = 0
	while prev != cost(theta):
		prev = cost(theta)
		
		theta = descent(theta)
		
		i+=1
		if i%100 == 0:
			print cost(theta)
	
	# print theta
	# print prev

	s = 0
	m = x2.shape[0]

	print theta
	
	for i in range(m):
		xi = x2[i:i+1].values[0]
		hyp = hypothesis(xi,theta)
		yi = y2[i:i+1].values[0]
		s += abs(absolute(hyp) - yi)
	
	print float(s)/float(m), m

