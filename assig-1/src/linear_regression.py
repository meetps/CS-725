import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets, linear_model

def read_training_data(input_path):
    raw_data = pd.read_csv(input_path, header=0)
    raw_data = raw_data._get_numeric_data()
    X = raw_data.as_matrix()[:, 0:-1]
    y = raw_data.as_matrix()[:, -1:]
    return X, y


def read_test_data(input_path):
    raw_data = pd.read_csv(input_path, header=0)
    raw_data = raw_data._get_numeric_data()
    X = raw_data.as_matrix()
    return X


def save_to_csv(preds, fname):
    pred_modif = preds
    pd.DataFrame({"id": list(range(0, len(preds))), "shares": pred_modif}).to_csv(fname, index=False, header=True)

train_data = "../data/train_data.csv"
test_data  = "../data/test_data.csv"

X_train, y_train = read_training_data(train_data)
X_test = read_test_data(test_data)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

preds = regr.predict(X_test)
print preds
save_to_csv(preds.reshape(7643), 'linear_regression_no_pp.csv')
