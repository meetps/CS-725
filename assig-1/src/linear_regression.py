import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets, linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def read_training_data(input_path):
    raw_data = pd.read_csv(input_path, header=0)
    raw_data = raw_data._get_numeric_data()
    X = MinMaxScaler().fit_transform(raw_data.as_matrix()[:, 0:-1])
    y = MinMaxScaler().fit_transform(raw_data.as_matrix()[:, -1:])
    y_mean = np.mean(raw_data.as_matrix()[:, -1:])
    return X, y, y_mean


def read_test_data(input_path):
    raw_data = pd.read_csv(input_path, header=0)
    raw_data = raw_data._get_numeric_data()
    X = MinMaxScaler().fit_transform(raw_data.as_matrix())
    return X


def save_to_csv(preds, fname, y_mean):
    pred_modif = preds * y_mean * 100
    for i in range(len(preds)):
        pred_modif[i] = int(round(pred_modif[i], -2))
    pd.DataFrame({"id": list(range(0, len(preds))), "shares": pred_modif}).to_csv(fname, index=False, header=True)

train_data = "../data/train_data.csv"
test_data  = "../data/test_data.csv"

X_train, y_train, y_mean = read_training_data(train_data)

pca = PCA(n_components=4)
X_train = pca.fit_transform(X_train)

X_test = read_test_data(test_data)
X_test = pca.fit_transform(X_test)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

preds = regr.predict(X_test)
print preds