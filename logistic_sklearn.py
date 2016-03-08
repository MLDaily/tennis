import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import scipy as sp

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

# Train classifier
clf = LogisticRegression()
clf.fit(x, y)

# Predict
y_pred = clf.predict_proba(x2)[:,1]

def absolute(z):
	if z > 0.5:
		return 1
	return 0

def sigmoid(z):
	z = 1/(1+np.power(e,-z))
	return z

m = len(y_pred)

s=0
for i in range(len(y_pred)):
	s += abs(absolute(y_pred[i]) - y2[i:i+1].values[0])

print 1-float(s)/float(m)
