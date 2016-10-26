import math
import numpy as np
import csv
import random
import math

def read_input():
	x_data = [];
	y_data = [];
	test = [];
	with open('../data/train.csv', 'rb') as csvfile:
		file = csv.reader(csvfile, delimiter=',', quotechar='|')
		init = 1
		for row in file:
			if(init == 1):
				init = 0
				continue
			temp = [float(i) for i in row]
			x_data.append(temp[0:10])
			y_data.append(temp[10])
	x_data = np.asarray(x_data)
	y_data = np.asarray(y_data)

	with open('../data/test1.csv', 'rb') as csvfile:
		file = csv.reader(csvfile, delimiter=',', quotechar='|')
		init = 1
		for row in file:
			if(init == 1):
				init = 0
				continue
			temp = [float(i) for i in row]
			test.append(temp)
	test = np.asarray(test)
	return x_data,y_data,test

def preprocess(X_in):
	X_out = np.zeros((X_in.shape[0],85))
	for i in range(X_in.shape[0]):
		for j in range(5):
			n = X_in[i,2*j + 1]
			s = X_in[i,2*j]
			X_out[i,17*j + int(n) - 1] = 1
			X_out[i,17*j + 12 + int(s)] = 1	
	return X_out

def one_hot_encoding(labels, num_classes=10):
    labels = labels.reshape((labels.shape[0],1))
    ycols = np.tile(labels, (1, num_classes))
    m, n = ycols.shape
    indices = np.tile(np.arange(num_classes).reshape((1,num_classes)), (m, 1))
    ymat = indices == ycols
    return ymat.astype(int)

def get_data(cv_split=0.99):
	[x_data,y_data,X_test] = read_input()
	train_split = int(x_data.shape[0]*cv_split)
	X_train = preprocess(x_data[0:train_split,:])
	y_train = y_data[0:train_split]
	X_valid = preprocess(x_data[train_split:x_data.shape[0],:])
	y_valid = y_data[train_split:x_data.shape[0]]
	return X_train, X_valid, y_train, y_valid, X_test