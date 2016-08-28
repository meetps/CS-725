import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets, linear_model

def pre_process(X_train, ratio=3.0):
    for i in range(len(X_train[0])):
        mean = np.mean(X_train, axis=0) 
        std =  np.std(X_train, axis=0)
        print mean, std
        for j in range(len(X_train)):
            if X_train[j][i] > mean  + ratio * std:
                X_train[j][i] = mean  + ratio * std
            else:
                pass
    return X_train


def read_training_data(input_path):
    raw_data = pd.read_csv(input_path, header=0)
    labels = list(pd.read_csv(input_path, nrows=1))
    X = raw_data.as_matrix()[:, 0:-1]
    y = raw_data.as_matrix()[:, -1:]
    return X, y, labels


def read_test_data(input_path):
    raw_data = pd.read_csv(input_path, header=0)
    raw_data = raw_data._get_numeric_data()
    X = raw_data.as_matrix()
    return X


def save_to_csv(preds, fname):
    pred_modif = preds
    for i in range(len(preds)):
        pred_modif[i] = int(round(pred_modif[i],-2))
    pd.DataFrame({"id": list(range(0, len(preds))), "shares": pred_modif}).to_csv(fname, index=False, header=True)

train_data = "../data/train_data.csv"
test_data  = "../data/test_data.csv"

X_train, y_train, labels = read_training_data(train_data)
X_test = read_test_data(test_data)

#Preprocessing 
X_train = pre_process(X_train)

for i in range(1,len(X_train[0])):
    plt.plot((X_train.T)[i], y_train)
    plt.title(labels[i])
    plt.show()

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets

# regr.fit(X_train, y_train)
# preds = regr.predict(X_test)
# print preds
# save_to_csv(preds.reshape(7643), 'linear_regression_no_pp.csv')
