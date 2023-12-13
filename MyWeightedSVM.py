import numpy as np

class MyWeightedSVM:

    def __init__(self, d, max_iters, eta_val, c, w1):
        self.w = np.zeros(d)
        self.w_old = np.random.uniform(-0.01, 0.01, d)
        self.w_sum = np.zeros(d)
        self.w_sum += self.w_old
        self.max_iters = max_iters
        self.eta_val = eta_val
        self.c = c
        self.iters = 0
        self.losses = []
        self.gradient_magnitudes = []
        self.w1 = w1

    def fit(self, X, y):
        (n, d) = X.shape
        while self.iters < self.max_iters:
            i = np.random.randint(n)
            class_weight = self.w1 if y[i] == 1 else 1
            loss = (1/2)*np.linalg.norm(self.w_old)**2 + self.c*class_weight*max(0, 1 - y[i]*(self.w_old @ X[i, :]))
            self.losses.append(loss)
            gradient_magnitude = 0
            for j in range(d):
                if y[i]*(self.w_old @ X[i, :]) < 1:
                    self.w[j] = self.w_old[j] - self.eta_val*(self.w_old[j] - self.c*class_weight*y[i]*X[i,j])
                    gradient_magnitude += (self.w_old[j] - self.c*class_weight*y[i]*X[i,j])**2
                else:
                    self.w[j] = self.w_old[j] - self.eta_val*(self.w_old[j])
                    gradient_magnitude += (self.w_old[j])**2
                self.w_old[j] = self.w[j]
            self.gradient_magnitudes.append(gradient_magnitude)
            self.w_sum += self.w
            self.iters += 1
            if np.average(self.gradient_magnitudes[-10:]) < 1e-6:
                break

    def predict(self, X):
        w_avg = self.w_sum / self.iters
        return np.sign(X @ w_avg)
    
    def predict_values(self, X):
        w_avg = self.w_sum / self.iters
        return X @ w_avg
    
    def predict_proportion(self, X, prop):
        w_avg = self.w_sum / self.iters
        values = X @ w_avg
        #print(values)
        threshold = np.quantile(values, 1-prop)
        #print("Threshold: ", threshold)
        preds = np.zeros(len(values))
        for i in range(len(preds)):
            if values[i] >= threshold:
                preds[i] = 1
            else:
                preds[i] = -1
        return preds
