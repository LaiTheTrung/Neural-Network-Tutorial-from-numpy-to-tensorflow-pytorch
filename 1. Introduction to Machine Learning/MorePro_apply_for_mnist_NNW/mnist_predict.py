from keras.datasets import mnist
import random
import numpy as np
from neural_Network1 import PredictNeuralNetWork
import matplotlib.pyplot as plt 
import os
(test_X, test_y), (test_X, test_y) = mnist.load_data()
def NumToVec(n, maxLength = 10):
	vec = np.zeros(10)
	vec[n] = 1
	vec = vec.reshape(1, maxLength)
	return vec
sub_test_X = []
sub_test_y = []
test_pair = list(zip(test_X, test_y))
random.shuffle(test_pair)
n_sample = 2000
for i in range(n_sample):
	x, y = test_pair[i]
	x = x.flatten()
	x = x.reshape(1, len(x))
	sub_test_X.append(x)
	y = NumToVec(y)
	sub_test_y.append(y)


Xs_test = np.array(sub_test_X).reshape(2000,784)
Ys_test = np.array(sub_test_y).reshape(2000,10)
print(Ys_test)



def take_data_weigth_tested():
	list_theta = []
	path = "./weight"
	all_files = os.listdir(path)
	for file_name in all_files:
		file_path = path + "/" + file_name
		theta = np.loadtxt(file_path)
		list_theta.append(theta)
	return list_theta

list_theta = take_data_weigth_tested()
y_estimated = PredictNeuralNetWork(Xs_test,list_theta).estimate()
wrongcase = (1/2)* np.sum(np.absolute(y_estimated-Ys_test))
accuracy = (1-wrongcase/n_sample )* 100
print(accuracy)
print(wrongcase)