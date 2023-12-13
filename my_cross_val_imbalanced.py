import numpy as np
import sklearn.metrics

# Metric: choice of evaluation metric (f1, precision, recall, etc.)
# Proportion: proportion of test set to predict as 1s, if needed
def my_cross_val_imbalanced(model, metric, proportion, X, y, k=10):
    (n, d) = X.shape
    validation_metrics = np.zeros(k)
    for i in range(k):
        val_set = X[round(i*n/k):round((i+1)*n/k), :]
        val_labels = y[round(i*n/k):round((i+1)*n/k)]
        train_set = np.delete(X, [j for j in range(round(i*n/k), round((i+1)*n/k))], 0)
        train_labels = np.delete(y, [j for j in range(round(i*n/k), round((i+1)*n/k))], 0)
        model.fit(train_set, train_labels)
        if proportion == None:
            y_preds = model.predict(val_set)
        else:
            y_preds = model.predict_proportion(val_set, proportion)
        
        tp, fp, tn, fn = 0, 0, 0, 0
        score = 0
        for j in range(len(y_preds)):
            if val_labels[j] == 1 and y_preds[j] == 1:
                tp += 1
            elif val_labels[j] == 1:
                fn += 1
            elif y_preds[j] == 1:
                fp += 1
            else:
                tn += 1
        if tp == 0: # to avoid division by zero error for trivial models
            precision = 0
            recall = 0
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)  
        if metric == 'precision':
            score = precision
        if metric == 'recall':
            score = recall
        if metric == 'f1':
            if precision + recall == 0:
                score = 0
            else:
                score = 2 * precision * recall / (precision + recall)
        if metric == 'auprc':
            score = sklearn.metrics.average_precision_score(val_labels, y_preds)
        validation_metrics[i] = score
    return validation_metrics

def precision_recall_f1(preds, truth):
    tp, fp, tn, fn = 0, 0, 0, 0
    for j in range(len(preds)):
        if truth[j] == 1 and preds[j] == 1:
            tp += 1
        elif truth[j] == 1:
            fn += 1
        elif preds[j] == 1:
            fp += 1
        else:
            tn += 1
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0
    recall = tp / (tp + fn)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else: 
        f1 = 0
    return (precision, recall, f1)
