import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.decomposition import RandomizedPCA
from sklearn import datasets, linear_model

def pre_process(X_train, y_train, ratio=(3.0,3.0)):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    for i in range(len(X_train[0])):
        print mean[i], std[i]
        for j in range(len(X_train)):
            if X_train[j][i] > mean[i] + ratio[0] * std[i] or X_train[j][i] < mean[i] - ratio[0] * std[i]:
                X_train[j][i] = mean[i] + ratio[0] * std[i]
            else:
                pass

    mean = np.mean(y_train, axis=0)
    std = np.std(y_train, axis=0)
    print mean, std
    for i in range(len(y_train)):
        if y_train[i] > mean + ratio[1] * std or y_train[i] < mean - ratio[1] * std :
            y_train[i] = mean + ratio[1] * std
        else:
            pass

    # pca = RandomizedPCA(X_train, whiten=False)
    # pca.fit(X_train)
    # X_train = pca.transform(X_train)
    return X_train, y_train


def read_training_data(input_path):
    raw_data = pd.read_csv(input_path, header=0)
    labels = list(pd.read_csv(input_path, nrows=1))
    X = raw_data.as_matrix()[:, 1:-1]
    X = X.astype(dtype='double')
    y = raw_data.as_matrix()[:, -1:]
    y = y.astype(dtype='double')
    return X, y, labels


def read_test_data(input_path):
    raw_data = pd.read_csv(input_path, header=0)
    raw_data = raw_data._get_numeric_data()
    X = raw_data.as_matrix()
    return X


def save_to_csv(preds, fname):
    pred_modif = preds
    # for i in range(len(preds)):
        # pred_modif[i] = int(round(pred_modif[i],-1))
    pd.DataFrame({"id": list(range(0, len(preds))), "shares": pred_modif}).to_csv(fname, index=False, header=True)

train_data = "../data/train_data.csv"
test_data  = "../data/test_data.csv"

X_train, y_train, labels = read_training_data(train_data)
X_test = read_test_data(test_data)

#Preprocessing
X_train, y_train = pre_process(X_train, y_train, ratio=(4.5, 2.0))

# PLotting
# plt.hist(y_train, bins=50)
# plt.show()

# for i in range(1,len(X_train[0])):
#     plt.hist((X_train.T)[i], bins=50)
#     plt.title(labels[i])
#     plt.show()

# Create linear regression object
# regr = linear_model.LinearRegression()
regr = linear_model.Ridge(alpha=0.5)
# regr = linear_model.Lasso(alpha=0.1)
# regr = linear_model.Lars(n_nonzero_coefs=np.inf)
# regr = linear_model.LarsCV(max_iter=1000, n_jobs=3)

# Train the model using the training sets

regr.fit(X_train, y_train)
preds = regr.predict(X_test)
print preds
save_to_csv(preds.reshape(7643), 'linear_regression_no_pp.csv')