import numpy as np
from copy import deepcopy

class GradientBooster:
    def __init__(self, model, rmodel, boosting_steps):
        self.model = model
        self.rmodel = rmodel
        self.steps = boosting_steps
        #self.models = []

        self.is_fitted = False
        return
    
    def fit(self, xtrain, ytrain):
        print('fitting...')
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.is_fitted = True

        print('fit model')
        self.model.fit(self.xtrain, self.ytrain)
        print('predict xtrain') 
        y_train_pred = self.model.predict(self.xtrain)
        print('xtrain done')

        for i in range(self.steps):
            print('fitting step', i)
            residuals = self.ytrain - y_train_pred
            self.rmodel.fit(xtrain, residuals)
            y_train_pred += self.rmodel.predict(self.xtrain)
            #self.models.append(deepcopy(self.rmodel))
        return
    
    def predict(self, xtest):
        print('predicting...')
        if not self.is_fitted:
            raise ValueError("is_fitted is false. Please fit the model.")
        
        ypred = self.model.predict(xtest)
        
        ypred = ypred - self.rmodel.predict(xtest)
        # for r in self.models:
        #     ypred += np.concatenate(r.predict(xtest))
        return ypred