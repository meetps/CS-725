import csv
import math
import numpy as np
import read_data as rd

from tqdm import *

learning_rate 		= 0.1
n_epoch       		= 1     # Increase to 25 for actual output
input_dim     		= 85
hidden_layer_dim    = 42
output_dim          = 10
cv_split            = 0.99

def predict(X_test):
	X_test = X_test[:,1:]
	test_input = rd.preprocess(X_test)
	ids = [row[0] for row in X_test]
	result = np.zeros(test_input.shape[0])
	for i in tqdm(range(test_input.shape[0])):
		[layer_1_test_output,layer_2_test_output]=(forward_pass(test_input[i,:],weights_1,weights_2))
		result[i] = (np.asarray(layer_2_output)).argmax()
	return result, ids

def save_csv(results, ids, filename="../output/output.csv"):
	np.savetxt(fname=filename,fmt="%d", X=zip(ids, results), delimiter=',', header="id,CLASS", comments="")

def forward_pass(input_vars,weights_1,weights_2):
	input_vars= np.asarray([1] + np.ndarray.tolist(input_vars))
	layer_1_out=list()
	for x in range(hidden_layer_dim):
		input_vars_w = weights_1[:,x]*input_vars
		input_vars_w = np.ndarray.tolist(input_vars_w)
		layer_1_out.append(layer_1[x].output(input_vars_w))

	layer_1_output=layer_1_out
	layer_1_out=[1]+layer_1_out
	layer_2_out=list()
	for x in range(output_dim):
		input_vars_w = weights_2[:,x]*layer_1_out
		input_vars_w = np.ndarray.tolist(input_vars_w)
		layer_2_out.append(layer_2[x].output(input_vars_w))
	layer_2_output=layer_2_out
	return layer_1_output,layer_2_output	


def back_prop(weights_1,weights_2,layer_1_output,layer_2_output,label,input_vars):

	nabla_w1=np.zeros((input_dim+1,hidden_layer_dim))
	nabla_w2=np.zeros((hidden_layer_dim+1,output_dim))

	nabla_sigma_2 = multiclass_cross_entropy(label)
	
	for i in range(output_dim):
		nabla_w2[0,i] = nabla_sigma_2[i]*layer_2_output[i]*(1-layer_2_output[i])
		for j in range(hidden_layer_dim):
			nabla_w2[j+1,i] = nabla_sigma_2[i]*layer_2_output[i]*(1-layer_2_output[i])*layer_1_output[j]
	nabla_sigma_1 = regularization(nabla_sigma_2,layer_2_output)
	
	for i in range(hidden_layer_dim):		
		nabla_w1[0,i]=nabla_sigma_1[i]*layer_1_output[i]*(1-layer_1_output[i])
		for j in range(input_dim):
			nabla_w1[j+1,i]=nabla_sigma_1[i]*layer_1_output[i]*(1-layer_1_output[i])*input_vars[j]
	return nabla_w1,nabla_w2

def multiclass_cross_entropy(label):
	nabla_sigma_2 = np.zeros(output_dim)
	for i in range(output_dim):
		if(i==label):
			nabla_sigma_2[i]= -1/layer_2_output[i]
		else:
			nabla_sigma_2[i]= 1/(1-layer_2_output[i])
	return nabla_sigma_2

def regularization(nabla_sigma_2,layer_2_output):
	nabla_sigma_1 = np.zeros(hidden_layer_dim)
	for i in range(hidden_layer_dim):
		temp = 0
		for j in range(output_dim):
			temp+=nabla_sigma_2[j]*layer_2_output[j]*(1-layer_2_output[j])*weights_2[i+1,j]
		nabla_sigma_1[i] = temp
	return nabla_sigma_1

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class neuron:
	def __init__(self,layer):
		self.layer=layer

	def output(self,weighted_input):
		self.out = sigmoid(sum(weighted_input))
		return self.out

print("="*50 ) 
print("Reading and Pre-processing data")

X_train, X_valid, Y, Y_valid, X_test = rd.get_data(cv_split)

layer_1 = list()
layer_2 = list()

for i in xrange(hidden_layer_dim):
	layer_1.append(neuron(1))
for i in xrange(output_dim):
	layer_2.append(neuron(2))


weights_1  = (np.random.rand(input_dim+1,hidden_layer_dim)*2/np.sqrt(input_dim)) -1/np.sqrt(input_dim)
weights_2  = (np.random.rand(hidden_layer_dim+1,output_dim))*2/np.sqrt(input_dim) - 1/np.sqrt(input_dim)
input_vars = [1]*input_dim
input_vars = [1] + input_vars
input_vars = np.asarray(input_vars)

for epoch in range(n_epoch):
	print("="*50 ) 
	print("Epoch " + str(epoch+1) + " started")		
	for i in tqdm(range(X_train.shape[0])):
		[layer_1_output,layer_2_output]=(forward_pass(X_train[i,:],weights_1,weights_2))
		[dweights_1,dweights_2] = back_prop(weights_1,weights_2,layer_1_output,layer_2_output,Y[i],X_train[i,:])	
		weights_1 = weights_1 - learning_rate*dweights_1
		weights_2 = weights_2 - learning_rate*dweights_2

print("="*50 ) 
print('Predicting values from trained neural network') 
results, ids = predict(X_test)
print("="*50 ) 
print('Generating submission csv file') 
save_csv(results, ids, "../output/output" + str(n_epoch) + "_epochs.csv")