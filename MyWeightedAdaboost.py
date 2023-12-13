import numpy as np
import copy

class MyWeightedAdaboost:
    
    def __init__(self, estimator, num_estimators, w1):
        
        self.estimator=estimator # just declaring the type of estimator! do not change it but make copies
        self.num_estimators=num_estimators
        self.classifiers = []
        self.alphas = []
        self.w1 = w1
        
    def fit(self, X, y):
        
        (n, d) = X.shape
        weights = np.array([(1/n) for i in range(n)])
        adaboost_predictions = np.zeros(n)
        
        # Initializing extra weight for ones
        one_indices = np.where(y==1)
        weights[one_indices] *= self.w1
        weights = weights/np.sum(weights)
        
        for t in range(self.num_estimators):
           
            estimator = copy.deepcopy(self.estimator)
            
            #generating dataset to fit from weight distribution
            samples = np.random.choice(np.array(range(n)), size=n, replace=True, p=weights)
            X_samp = X[samples]
            y_samp = y[samples]
            
            estimator.fit(X_samp, y_samp)
            #print(np.argmax(estimator.feature_importances_)) # prints most important column, can use to check dataframe and see column name
            predictions = estimator.predict(X)
            
            error = 0
            for i in range(n):
                if predictions[i] != y[i]:
                    error += weights[i] 

            alpha = (1/2)*np.log((1 - error)/error)
            
            for i in range(n):
                if predictions[i] != y[i]: 
                    weights[i] *= np.exp(alpha)
                else:
                    weights[i] *= np.exp(-alpha)
            
            #Normalize weights
            weights = weights/np.sum(weights)
            
            self.classifiers.append(estimator)
            self.alphas.append(alpha)
        
    def predict(self, X):
        (n, d) = X.shape
        predictions = np.zeros(n)
        for t in range(self.num_estimators):
            predictions += self.alphas[t] * self.classifiers[t].predict(X)
        #print(predictions)
        return np.sign(predictions)
    
    def predict_values(self, X):
        (n, d) = X.shape
        predictions = np.zeros(n)
        for t in range(self.num_estimators):
            predictions += self.alphas[t] * self.classifiers[t].predict(X)
        #print(predictions)
        return predictions
    
    def predict_proportion(self, X, prop):
        (n, d) = X.shape
        predictions = np.zeros(n)
        for t in range(self.num_estimators):
            predictions += self.alphas[t] * self.classifiers[t].predict(X)
        threshold = np.quantile(predictions, 1-prop)
        preds = np.zeros(n)
        for i in range(len(preds)):
            if predictions[i] >= threshold:
                preds[i] = 1
            else:
                preds[i] = -1
        return preds