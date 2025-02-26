import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.decomposition import RandomizedPCA
from sklearn import datasets, linear_model


def pre_process_test(X_test,ratio=3.0):
    mean = np.mean(X_test, axis=0)
    std = np.std(X_test, axis=0)
    for i in range(len(X_test[0])):
        # print mean[i], std[i]
        for j in range(len(X_test)):
            if X_test[j][i] > mean[i] + ratio * std[i]:
                X_test[j][i] = mean[i] + ratio * std[i]
            elif X_test[j][i] < mean[i] - ratio * std[i]:
                X_test[j][i] = mean[i] - ratio * std[i]
            else:
                pass
    return X_test


def pre_process(X_train, y_train, ratio=(3.0,3.0)):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    for i in range(len(X_train[0])):
        # print mean[i], std[i]
        for j in range(len(X_train)):
            if X_train[j][i] > mean[i] + ratio[0] * std[i]:
                X_train[j][i] = mean[i] + ratio[0] * std[i]
            elif X_train[j][i] < mean[i] - ratio[0] * std[i]:
                X_train[j][i] = mean[i] - ratio[0] * std[i]
            else:
                pass

    mean = np.mean(y_train, axis=0)
    std = np.std(y_train, axis=0)
    # print mean, std
    for i in range(len(y_train)):
        if y_train[i] > mean + ratio[1] * std:
            y_train[i] = mean + ratio[1] * std
        elif y_train[i] < mean - ratio[1] * std:
            y_train[i] = mean - ratio[1] * std
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
    X = X.astype(dtype='float')
    y = raw_data.as_matrix()[:, -1:]
    y = y.astype(dtype='float')
    return X, y, labels


def read_test_data(input_path):
    raw_data = pd.read_csv(input_path, header=0)
    X = raw_data.as_matrix()[:, 1:]
    X = X.astype(dtype='float')
    return X


def plot(X_train, y_train):
    plt.hist(y_train, bins=50)
    plt.show()
    for i in range(1, len(X_train[0])):
        plt.hist((X_train.T)[i], bins=50)
        plt.title(labels[i])
        plt.show()


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def save_to_csv(preds, fname):
    pred_modif = preds
    for i in range(len(preds)):
        pred_modif[i] = int(round(pred_modif[i],-2))
    pd.DataFrame({"id": list(range(0, len(preds))), "shares": pred_modif}).to_csv(fname, index=False, header=True)

train_data = "data.csv"
test_data  = "data.csv"

X_train, y_train, labels = read_training_data(train_data)
# X_test = read_test_data(test_data)

# #Statistical Correlation
# for i in range(len(labels)-2):
#     corr = np.correlate(X_train.T[i,:], y_train.T[0,:])
#     print i, labels[i+1], corr

# # Delete low corr columns 
# low_corr_index = [3,4,5,12,13,14,15,16,17,30,31,32,33,34,37,52,53,54]
# # for i in range(len(low_corr_index)):
# X_train = np.delete(X_train, low_corr_index, 1)
# X_test = np.delete(X_test, low_corr_index, 1)

# #Preprocessing
# X_train, y_train = pre_process(X_train, y_train, ratio=(3.0, 1.30))
# X_test = pre_process_test(X_test, ratio=3.0)

# # PLotting
# # plot(X_train, y_train)

# # Create linear regression object
# # regr = linear_model.LinearRegression()
# regr = linear_model.Ridge(alpha=0.5)
# # regr = linear_model.Lasso(alpha=0.1)
# # regr = linear_model.Lars(n_nonzero_coefs=np.inf)
# # regr = linear_model.LarsCV(max_iter=1000, n_jobs=3)

# # Train the model using the training sets
# cv_fold = 10
# regr.fit(X_train[len(X_train)/cv_fold:], y_train[len(X_train)/cv_fold:])

# # Cross-Validate
# cv_preds = regr.predict(X_train[:len(X_train)/cv_fold])
# mse = rmse(y_train[:len(X_train)/cv_fold], cv_preds)
# print mse

# # Predict the output from trained model
# preds = regr.predict(X_test)
# print preds
# save_to_csv(preds.reshape(7643), 'linear_regression_no_pp.csv')
