# dependencies
import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse

from scipy.spatial.distance import cdist

def Quartic(x):
  return np.where(np.abs(x)>1,0,15/16*(1-np.abs(x)**2)**2)

def Tricubic(x):
  return np.where(np.abs(x)>1,0,(1-np.abs(x)**3)**3)

def Gaussian(x):
  return np.where(np.abs(x)>4,0,1/(np.sqrt(2*np.pi))*np.exp(-1/2*x**2))

class LowessWithcdist:
    '''Locally weighted prediction using scipy's cdist.'''
    def __init__(self, kernel=Quartic, tau=0.01):
        # assign the lowess model's kernel and value for tau (the width of the band)
        self.kernel = kernel 
        self.tau = tau
        return
    
    def fit(self, xtrain, ytrain):
        # fit the model on given training values
        self.xtrain = xtrain
        self.ytrain = ytrain
        return
    
    def dist(self, xtest):
        # given something for x2 and the xtrain in the class, calculate their distance
        return cdist(self.xtrain, xtest, metric='Euclidean')
    
    def calculate_weights(self, calculated_distance):
        # using previously calculated distance, return weights
        return self.kernel((calculated_distance)/(2*self.tau))
    
    def predict(self, xtest, lm=linear_model.Ridge(alpha = 0.001)):
        # after the model is fit, predict values
        # model is an argument so alpha can be maniuplated more simply

        distance = self.dist(xtest)
        weights = self.calculate_weights(distance)

        ypred = []
        for i in range(len(xtest)):
            # for each value in xtest, fit the model to the ith column of weights @ xtrain, and the same @ ytrain (?)
            lm.fit(np.diag(weights[:,i])@self.xtrain,np.diag(weights[:,i])@self.ytrain)
            ypred.append(lm.predict(xtest[i].reshape(1,-1)))
        return ypred