import numpy as np
import pandas as pd
import sklearn.metrics 
import matplotlib.pyplot as plt

from my_cross_val_imbalanced import my_cross_val_imbalanced, precision_recall_f1
from MyWeightedSVM import MyWeightedSVM

#PREPROCESSING
train_data = pd.read_csv('data/aug_train.csv')
test_data = pd.read_csv('data/aug_test.csv')
y_test = np.load('data/answer.npy')
train_samples = len(train_data)
test_samples = len(test_data)
data = pd.concat([train_data, test_data])
data = pd.get_dummies(data, columns=['Gender', 'Region_Code', 'Vehicle_Age', 'Vehicle_Damage', 'Policy_Sales_Channel'])

data.loc[data['Response'] == 0, 'Response'] = -1 # changing negative class to -1 for svm to work

train_data = data.iloc[0:train_samples]
test_data = data.iloc[train_samples:]
test_data = test_data.drop('Response', axis=1)

train_data = train_data.drop('id', axis=1)
test_data = test_data.drop('id', axis=1)

X_train = train_data.drop('Response', axis=1)
y_train = train_data['Response']
X_test = test_data

X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()

#TRAINING
#SVM with proportion strategy
eta_vals = [0.00001, 0.0001, 0.001]
C_vals = [1, 10, 100]
weights = [1, 2, 5, 10, 20, 50]
best_score = 0
best_eta, best_c, best_weight = (0, 0, 0)

(_, num_features) = X_train.shape
proportion = len(y_train[y_train == 1])/len(y_train)

for eta_val in eta_vals:
    for c_val in C_vals:
        for w in weights:

            # instantiate svm object
            svm = MyWeightedSVM(num_features, 100000, eta_val, c_val, w)

            # call to your CV function to compute error rates for each fold
            cv_scores = my_cross_val_imbalanced(svm, 'f1', proportion, X_train, y_train, k=10)
            #cv_scores = my_cross_val_imbalanced(svm, 'f1', None, X_train, y_train, k=10)

            # print error rates from CV
            print("Eta: " + str(eta_val))
            print("C: " + str(c_val))
            print("Weight: " + str(w))
            for i in range(10):
                print("F1 score for fold " + str(i) + ": " + str(cv_scores[i]))
            mean_score = sum(cv_scores)/len(cv_scores)
            print("Mean validation F1 score: " + str(mean_score))
            print("Validation F1 score stdev: " + str(np.std(cv_scores)))
            if mean_score >= best_score:
                best_score = mean_score
                best_eta, best_c, best_weight = (eta_val, c_val, w)

# instantiate svm object for best value of eta and C
print("Best eta value: " + str(best_eta))
print("Best C value: " + str(best_c))
print("Best weight: " + str(best_weight))
best_svm = MyWeightedSVM(num_features, 100000, best_eta, best_c, best_weight)

# fit model using all training data
best_svm.fit(X_train, y_train)

# predict on test data
y_preds = best_svm.predict_proportion(X_test, proportion)
#y_preds = best_svm.predict(X_test)
y_values = best_svm.predict_values(X_test)

# compute F1 score on test data
(precision, recall, f1) = precision_recall_f1(y_preds, y_test)
auprc = sklearn.metrics.average_precision_score(y_test, y_values)

print("Test precision: " + str(precision))
print("Test recall: " + str(recall))
print("Test F1 score: " + str(f1))
print("Test AUPRC score: " + str(auprc)) 

display = sklearn.metrics.PrecisionRecallDisplay.from_predictions(y_test, y_values, name="SVM")
display.plot()
plt.show()