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
		# print l
		l = np.multiply(alpha,l)
	
	theta = np.subtract(theta,l)

	return theta

if __name__ == '__main__':
	theta = np.zeros(x.shape[1])
	prev = cost(theta)
	print prev
	theta = descent(theta)

	i = 0
	while prev != cost(theta):
		prev = cost(theta)
		theta = descent(theta)
		if i%100 == 0:
			print cost(theta)
		if prev > cost(theta):
			break
		i+=1

	print theta
		
	s = 0
	m = x2.shape[0]

	for i in range(m):
		xi = x2[i:i+1].values[0]
		hyp = hypothesis(xi,theta)
		yi = y2[i:i+1].values[0]
		s += abs(absolute(hyp) - yi)
	
	print float(s)/float(m), m

# Output
# [-3.39720911]
# [-3.19059474]
# [-3.08936486]
# [-3.02405067]
# [-2.97716602]
# [-2.9415678]
# [-2.91360774]
# [-2.89115477]
# [-2.87284621]
# [-2.85775227]
# [-2.84520717]
# [-2.83471637]
# [-2.82590201]
# [-2.81846891]
# [-2.81218247]
# [-2.80685365]
# [-2.80232847]
# [-2.79848033]
# [-2.79520446]
# [-2.79241361]
# [-2.79003475]
# [-2.78800652]
# [-2.7862772]
# [-2.78480302]
# [-2.7835469]
# [-2.78247731]
# [-2.78156741]
# [-2.7807943]
# [-2.78013843]
# [-2.77958306]
# [-2.77911385]
# [-2.77871852]
# [-2.7783865]
# [-2.77810872]
# [-2.77787739]
# [-2.77768576]
# [-2.77752807]
# [-2.7773993]
# [-2.77729515]
# [-2.7772119]
# [-2.77714632]
# [-2.77709564]
# [-2.77705745]
# [-2.77702967]
# [-2.77701049]
# [-2.77699834]
# [-2.77699189]
# [-2.77698994]
# [ -1.42872123e-002   1.07714910e-002  -8.25106676e-003  -8.48928193e-002
#    1.58901757e-001  -6.72549948e-002  -1.29958147e-003   5.82587256e-002
#   -1.33302320e-002   1.26677507e-001  -3.38062488e-002  -1.28232551e-002
#   -9.62325856e-003  -6.66113877e-002  -4.03374134e-003   1.41936146e-185
#    4.68151465e-003  -1.11010940e-002   4.63992450e-003   2.12545913e-002
#    2.12606720e-002  -1.12085523e-001   4.67060156e-002  -4.00622912e-002
#    6.82947390e-002  -3.15761154e-002   4.14141306e-003  -1.25168384e-002
#    2.49472297e-002  -6.20073607e-002  -2.86021657e-002  -5.65789575e-002
#   -4.33323930e-002   2.16996574e-186  -1.26723292e-002  -1.43349189e-002
#   -9.27203500e-003   2.47919392e-002   1.06303012e-002]
# 0.425531914894 94
