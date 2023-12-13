import numpy as np

def sigmoid(a):
    return 1 / (1 + np.exp(-a))
def L(y, x, w, weight): # weighted log loss function
    return - (weight * y * np.log(sigmoid(w @ x)) + (1 - y)*np.log(sigmoid(-w @ x)))

class MyWeightedLogisticRegression:

    def __init__(self, d, max_iters, eta_val, weight):
        self.w = np.zeros(d)
        self.w_old = np.random.uniform(-0.01, 0.01, d)
        self.w_sum = np.zeros(d)
        self.w_sum += self.w_old
        self.max_iters = max_iters
        self.eta_val = eta_val
        self.weight = weight
        self.iters = 0 # keep track of iterations made
        self.losses = [] # used to keep track of losses for plotting purposes
        self.gradient_magnitudes = [] 
    def fit(self, X, y):
        (n, d) = X.shape
        while self.iters < self.max_iters:
            i = np.random.randint(n)
            z = sigmoid(X @ self.w_old)
            self.losses.append(L(z[i], X[i, :], self.w_old, self.weight))
            gradient_magnitude = 0
            for j in range(d):
                gradient_j = - (y[i]*X[i, j]*(self.weight*sigmoid(-self.w_old @ X[i, :]) + sigmoid(self.w_old @ X[i, :])) - X[i, j] * sigmoid(self.w_old @ X[i, :]))
                self.w[j] = self.w_old[j] - self.eta_val*gradient_j
                gradient_magnitude += gradient_j**2
                self.w_old[j] = self.w[j]
            self.gradient_magnitudes.append(gradient_magnitude)
            self.w_sum += self.w
            self.iters += 1
            if np.average(self.gradient_magnitudes[-10:]) < 1e-7:
                break
    def predict(self, X):
        w_avg = self.w_sum / self.iters
        return np.round(sigmoid(X @ w_avg))
    def predict_probs(self, X):
        w_avg = self.w_sum / self.iters
        return sigmoid(X @ w_avg)
    def predict_proportion(self, X, prop): #predict a certain number of 1s
        w_avg = self.w_sum / self.iters
        probs = sigmoid(X @ w_avg)
        #print(probs)
        threshold = np.quantile(probs, 1-prop)
        #print("Threshold: ", threshold)
        preds = np.zeros(len(probs))
        for i in range(len(preds)):
            if probs[i] >= threshold:
                preds[i] = 1
        return preds
