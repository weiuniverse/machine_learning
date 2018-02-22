import numpy as np
import random
from scipy.sparse import *


class lasso:
    '''
    implementation of lasso algorithm
    solving : ...

    method:
        l = lasso(lamb)
        l.fit(X,y)
        y_pred = l.predict(X)

    '''
    def __init__(self,init_method="normal"):
        '''

        init_method = "uniform", "normal"

        train_mode = "fixed", "decay"
        '''

        if init_method == "uniform" or init_method =="normal":
            self.init_method = init_method
        else:
            print("Error: init_method is uniform or normal")


    def __init_param(self):
        '''
        "unifrom"
        "normal"
        '''
        d = self.__dimension
        if self.init_method == "uniform":
            self.__wt = np.random.uniform(-1,1(1,d))
            self.__bias = np.random.uniform(-1,1)
        elif self.init_method == "normal":
            self.__wt = np.random.normal(0,1,size=(1,d))
            self.__bias = np.random.normal(0,1)
        return self.__wt,self.__bias

    def fit(self,X,y,lamb,init=True):
        '''
        input:
        X is the data, a d * n array, d is the dimension of every data, n is the ammount of the data
        y is the label,a 1 * n array.
        lambda is a hyperparameter in regularization part.

        '''
        if lamb < 0:
            print("Error: lamb should be >=0")
        # get the shape of input
        d,n = X.shape
        self.__dimension = d

        # change X to csr format sparse matrix
        if issparse(X)==False:
            X = csr_matrix(X)
        # cal a
        a = 2 * (np.sum(X.multiply(X), axis = 1))

        # initial the parameters
        if init:
            wt,b = self.__init_param()
            wt = csr_matrix(wt)
        else:
            wt,b = self.get_param()
            wt = csr_matrix(wt)
        # initial delta, delta_wt
        delta = np.inf
        delta_wt = 10
        iters = 0
        while delta >0.0001:
            # adjust parameter
            print("train_iterations:%d :%f " % (iters,delta))
            wt_pre = wt.copy()
            # recal r
            r = csr_matrix(y - (wt.dot(X).todense()+b))
            # calculate objective value before update
            obj_val_pre = np.sum(r.multiply(r)) + lamb*np.sum(abs(wt))
            # update b
            b = b + np.mean(r)
            # update r
            # r = csr_matrix(y-(wt.dot(X).todense()+b))
            r = csr_matrix((r.todense()) - np.mean(r))
            for k in range(d):
                wt_k_pre = wt[:,k]
                c = 2*np.sum(X[k,:].dot(r.T))+a[k]*wt[:,k]
                # print(c,lamb)
                # update w
                if c< -lamb:
                    wt[:,k]=(c+lamb)/a[k]
                elif c>lamb:
                    wt[:,k]=(c-lamb)/a[k]
                else:
                    wt[:,k]=0

                # update r
                r = csr_matrix(y-(wt.dot(X).todense()+b))
                # r = r + (wt_k_pre-wt[:,k])*X[k,:]
            # cal delta_wt
            obj_val = np.sum(r.multiply(r)) + lamb*np.sum(abs(wt))
            delta = obj_val_pre - obj_val
            delta_wt = np.sum(abs(wt_pre-wt))
            iters = iters + 1
        self.__wt = np.array(wt.todense())
        self.__bias = b

    def predict(self,X):
        '''
        input: X d*n array
        output: y 1*n array
        '''
        if issparse(X)==False:
            X = csr_matrix(X)
        wt = csr_matrix(self.__wt)
        b = self.__bias
        y = (wt.dot(X)).todense() + b
        y = np.array(y)
        return y


    def get_param(self):
        '''
        output:
            parameters: wt(1*d array), b(scalar)
        '''
        return [self.__wt,self.__bias]

    def check_anwser(self,X,y):
        wt = csr_matrix(self.__wt)
        b = self.__bias
        # vec = 2*np.dot((np.dot(wt,X)+b-y),X.T)
        vec_tmp = csr_matrix(wt.dot(X).todense()+b-y)
        vec = 2*vec_tmp.dot(X.T)
        return vec

    def save_model(self,save_file):
        import pickle
        with open(save_file,"wb") as file:
            pickle.dump([self.__wt,self.__bias],file)


    def load_model(self,load_file):
        import pickle
        with open(load_file,"rb") as file:
            param = pickle.load(file)
        self.__wt = param[0]
        self.__bias = param[1]
        return self.__wt,self.__bias

def max_lamb(X,y):
    '''
    max_lamb = ||X(y-mean(y)).T||^inf
    '''
    # lamb = np.max(abs(np.dot(X,(y-np.mean(y)).T)))
    if issparse(X)==False:
        X = csr_matrix(X)
    lamb = 2*np.max(X.dot((y-np.mean(y)).T))
    return lamb

def RMSE(y_pred,y):
    error = np.mean((y_pred-y)**2)
    return error


'''
vec = 2*np.dot((np.dot(wt,X)+b-y),X.T)
print(np.sum(vec))
'''
