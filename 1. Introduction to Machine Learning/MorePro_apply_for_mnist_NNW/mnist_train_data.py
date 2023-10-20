from keras.datasets import mnist
import random
import numpy as np
from neural_Network1 import NeuralNetWork
import os 

(train_X, train_y), (test_X, test_y) = mnist.load_data()
##################################################
#One hot encoder
def NumToVec(n, maxLength = 10):
	vec = np.zeros(10)
	vec[n] = 1
	vec = vec.reshape(1, maxLength)
	return vec
##################################################
sub_train_X = []
sub_train_y = []

train_pair = list(zip(train_X, train_y))
random.shuffle(train_pair)
n_sample = 2000
for i in range(n_sample):
	x, y = train_pair[i]
	x = x.flatten()
	x = x.reshape(1, len(x))
	sub_train_X.append(x)
	y = NumToVec(y)
	sub_train_y.append(y)
############# c

#################################### Data preparation
Xs_train = np.array(sub_train_X).reshape(2000,784)
Ys_train = np.array(sub_train_y).reshape(2000,10)

lr=0.00035
rl=0.001
hiddenlayer = [81,81]
epochs = 500
K=10
NNW = NeuralNetWork(Xs_train,Ys_train,K,lr,rl,hiddenlayer)
NNW.init_theta()
Acc,Losses = NNW.train(epochs)
list_data_weight_trained = NNW.list_theta
##################################################
def delete_curent_data_in_weigth_folder():
	path = "./weight"
	all_files = os.listdir(path)
	for file_name in all_files:
		file_path = path + "/" + file_name
		os.remove(file_path)

def update_new_data_to_weigth_folder(list_data):
	for i,data in enumerate(list_data):
		name_saved_weith = "weight/weight_" + str(i)
		np.savetxt(name_saved_weith,data,fmt='%.150f')

##################################################
delete_curent_data_in_weigth_folder ()
update_new_data_to_weigth_folder (list_data_weight_trained)

