from lasso import max_lamb
from lasso import lasso
import numpy as np
from scipy.sparse import *
import matplotlib.pyplot as plt

def precision_recall(wt_pred):
    wt_pred = wt_pred[0,:]
    t = np.count_nonzero(wt_pred[0:10])
    f = np.count_nonzero(wt_pred[10:])
    if t+f==0:
        precision = 1
        recall = 0
    else:
        precision = t/(t+f)
        recall = t/10
    return precision,recall

## sample input
d = 80
n = 250
k = 10
wt = np.zeros((1,d))
wt[:,0:5] = -10
wt[:,5:k] = 10
b = 0
mu = 0
sigma = 1
noise = np.random.normal(mu,sigma,(1,n))

X = np.random.normal(0,1,(d,n))
y = np.dot(wt,X) + b + noise

X = csr_matrix(X)


### plot the precision recall - lambda
l_syn = lasso()
lamb = max_lamb(X,y)
print(lamb)
precision_list = []
recall_list = []
lamb_list = []
for i in range(10):
    l_syn.fit(X,y,lamb)
    wt_pred,b = l_syn.get_param()
    p,r = precision_recall(wt_pred)
    precision_list.append(p)
    recall_list.append(r)
    lamb_list.append(lamb)
    lamb = lamb/2

print(lamb_list)

plt.figure(1)
plt.title('precision_recall - lambda')
plt.xlabel('lambda')
plt.ylabel('precision & recall')

plt.plot(lamb_list,precision_list,label='precision')
plt.plot(lamb_list,recall_list,label='recall')
plt.legend()
plt.show()
