import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def read_training_data(input_path):
    raw_data = pd.read_csv(input_path, header=0)
    raw_data = raw_data._get_numeric_data()
    X = MinMaxScaler().fit_transform(raw_data.as_matrix()[:, 1:-1])
    y = raw_data.as_matrix()[:, -1:] / 100000.0
    y_mean = np.mean(raw_data.as_matrix()[:, -1:])
    # pca = PCA(n_components=5)
    # return pca.fit_transform(X), y, y_mean
    return X, y, y_mean


def read_test_data(input_path):
    raw_data = pd.read_csv(input_path, header=0)
    raw_data = raw_data._get_numeric_data()
    X = MinMaxScaler().fit_transform(raw_data.as_matrix())[:,1:]
    # pca = PCA(n_components=5)
    # return pca.fit_transform(X)
    return X


def save_to_csv(preds, fname, y_mean):
    pred_modif = preds * 100000.0
    for i in range(len(preds)):
        pred_modif[i] = int(round(pred_modif[i], -1))
    pd.DataFrame({"id": list(range(0, len(preds))), "shares": pred_modif}).to_csv(fname, index=False, header=True)


def baseline_model():
    model = Sequential()
    model.add(Dense(400, input_dim=58, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


train_data = "../data/train_data.csv"
test_data  = "../data/test_data.csv"

X_train, y_train, y_mean = read_training_data(train_data)
X_test = read_test_data(test_data)

batch_size = 50
nb_epoch = 25

regressor = baseline_model()
regressor.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)

score = regressor.evaluate(X_train, y_train, batch_size=batch_size, verbose=1)
print "Results:", score

preds = regressor.predict(X_test, batch_size=batch_size)
print preds, y_mean
save_to_csv(preds.flatten(), "keras_pure_val_20.csv", y_mean)
