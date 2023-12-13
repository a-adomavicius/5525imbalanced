import numpy as np
import pandas as pd
import sklearn.metrics 
import matplotlib.pyplot as plt

from my_cross_val_imbalanced import my_cross_val_imbalanced, precision_recall_f1
from MyWeightedLogisticRegression import MyWeightedLogisticRegression

#PREPROCESSING
train_data = pd.read_csv('data/aug_train.csv')
test_data = pd.read_csv('data/aug_test.csv')
y_test = np.load('data/answer.npy')
train_samples = len(train_data)
test_samples = len(test_data)
data = pd.concat([train_data, test_data])
data = pd.get_dummies(data, columns=['Gender', 'Region_Code', 'Vehicle_Age', 'Vehicle_Damage', 'Policy_Sales_Channel'])

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
# Weighted Logistic Regression with predicting fixed proportion of 1s 
eta_vals = [0.001, 0.01, 0.1]
weight_vals = [1, 2, 5, 10, 20, 50]

best_score = 0
best_eta = 0
best_weight = 0

(_, num_features) = X_train.shape
proportion = sum(y_train)/len(y_train)

# Logistic Regression
for eta_val in eta_vals:
    for weight in weight_vals:

        # instantiate logistic regression object
        lr = MyWeightedLogisticRegression(num_features, 1000000, eta_val, weight)

        # call to your CV function to compute error rates for each fold
        cv_scores = my_cross_val_imbalanced(lr, 'f1', proportion, X_train, y_train, k=10)
        #cv_scores = my_cross_val_imbalanced(lr, 'auprc', None, X_train, y_train, k=10)

        # print error rates from CV
        print("Eta: " + str(eta_val))
        print("Weight: " + str(weight))
        for i in range(10):
            print("F1 score for fold " + str(i) + ": " + str(cv_scores[i]))
        mean_score = sum(cv_scores)/len(cv_scores)
        print("Mean validation F1 score: " + str(mean_score))
        print("Validation F1 score stdev: " + str(np.std(cv_scores)))
        if mean_score >= best_score:
            best_score = mean_score
            best_eta = eta_val
            best_weight = weight

# instantiate logistic regression object for best value of eta
print("Best eta value: " + str(best_eta))
print("Best weight value: " + str(best_weight))
best_lr = MyWeightedLogisticRegression(num_features, 1000000, best_eta, best_weight)

# fit model using all training data
best_lr.fit(X_train, y_train)

# predict on test data
y_preds = best_lr.predict_proportion(X_test, proportion)
#y_preds = best_lr.predict(X_test)
print("Number of positive predictions: ", sum(y_preds))
y_values = best_lr.predict_values(X_test)

# compute F1 score on test data
(precision, recall, f1) = precision_recall_f1(y_preds, y_test)
auprc = sklearn.metrics.average_precision_score(y_test, y_values)

print("Test precision: " + str(precision))
print("Test recall: " + str(recall))
print("Test F1 score: " + str(f1))
print("Test AUPRC score: " + str(auprc)) 

display = sklearn.metrics.PrecisionRecallDisplay.from_predictions(y_test, y_values, name="LR")
display.plot()
plt.show()