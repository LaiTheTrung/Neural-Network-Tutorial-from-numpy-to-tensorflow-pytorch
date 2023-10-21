import numpy as np
import matplotlib.pyplot as plt
class ShallowNeuralNet():
	def __init__(self, Xs, Ys, input_size, hidden_size, output_size):
		self.Xs = Xs #(2000, 784) List(nparray) for example with 2000 is the number of samples
		self.Ys = Ys #(2000, 10) List(nparray) for example
		self.input_size = input_size # 784 for example
		self.hidden_size = hidden_size # 81 for example
		self.output_size = output_size # 10 for example
		self.w1 = self.GenerateWeight(input_size, hidden_size) #(785, 81)
		self.w2 = self.GenerateWeight(hidden_size, output_size) #(82, 10)

	def GenerateWeight(self, input_size, output_size):
		return np.random.rand(output_size, input_size+1).transpose() -0.5 # range [-0.5;0.5)

	def Forward(self, x, w1, w2):
		a1 = x #(1,785)
		a1_bias = self.AddBias(a1)
		z2 = np.matmul(a1_bias, w1) #(1, 785) x (785, 81) => (1,81)
		a2 = self.sigmoid(z2) #(1,81)
		a2_bias = self.AddBias(a2) #(1,82)
		z3 = np.matmul(a2_bias, w2) #(1,82) x (82, 10) => (1,10)
		a3 = self.sigmoid(z3)
		return a3

	def sigmoid(self, z):
		return 1/(1+np.exp(-z))

	def AddBias(self, input):#update new value for weight
		cur_len = len(input.flatten())
		return np.concatenate(([1], input.flatten())).reshape(1,cur_len+1)

	def Backward(self, x, y, w1, w2):
		a1 = x #(1,785)
		a1_bias = self.AddBias(a1)
		z2 = np.matmul(a1_bias, w1) #(1, 785) x (785, 81) => (1,81)
		a2 = self.sigmoid(z2) #(1,81)
		a2_bias = self.AddBias(a2) #(1,82)
		z3 = np.matmul(a2_bias, w2) #(1,82) x (82, 10) => (1,10)
		a3 = self.sigmoid(z3) #(1,10)

		#calculate delta
		d3 = a3 - y #(1,10)
		d2 = (np.matmul(d3, w2[1:,:].transpose())) * (a2*(1-a2)) #(1,10) x (82, 10).T => (1,82) (1,81+1)

		delta1 = np.matmul(a1_bias.transpose(), d2) #(1,785).T x (1,82) => (785,82) 
		delta2 = np.matmul(a2_bias.transpose(), d3) #(1,82).T x (1,10) => (82,10)
		return delta1, delta2

	def MSE(self, pred, y):
		return np.mean(np.power(pred - y,2))
	
	def train(self, epochs, lr):
		acc = []
		losses = []
		for epoch in range(epochs):
			cap_delta1 = 0
			cap_delta2 = 0
			l = []
			correct_case = 0
			for i in range(len(self.Xs)):
				x = self.Xs[i]
				y = self.Ys[i]
				pred = self.Forward(x, self.w1, self.w2)
				if pred.argmax() == y.argmax():
					correct_case+=1
				l.append(self.MSE(pred, y))
				delta1, delta2 = self.Backward(x, y, self.w1, self.w2)
				cap_delta1+= delta1
				cap_delta2+= delta2
			self.w1 = self.w1 - (lr*(cap_delta1))
			self.w2 = self.w2 - (lr*(cap_delta2))
			cur_acc = (correct_case/len(self.Xs))*100
			acc.append(cur_acc)
			mean_loss = np.array(l).mean()
			losses.append(mean_loss)
			freq = 5
			if (epoch+1) % freq == 0:
				avg_acc = np.array(acc[-5:]).mean()
				_loss = np.array(losses[-5:]).mean()
				print(f"Epoch {epoch + 1}: AVG ACC during {freq} epochs: {avg_acc}")
				print(f"Epoch {epoch + 1}: Loss: {_loss}")
		return acc, losses
	

class Multi_layer_NeuralNet:
	def __init__(self,Xs,Ys,list_of_hiddenNeural_in_layers):
		# m is number of sample, n is number of inputs, k is number of class
		self.X = Xs 
		self.y = Ys 
		self.L = len(list_of_hiddenNeural_in_layers) + 2

		self.m,self.n = Xs.shape
		_,self.k = Ys.shape

		self.Weight = []
		
		# intial weight
		unit_in_layer = [self.n] + list_of_hiddenNeural_in_layers + [self.k]
		# print('unit in layer: ',unit_in_layer)
		for l in  range(self.L-1): 
			weight = self.GenerateWeight(unit_in_layer[l],unit_in_layer[l+1]) #s(l+1)x(sl+1)
			print(f"sample weight {l+1}: {weight.shape}")
			self.Weight.append(weight)
		print(len(self.Weight))
		print(self.Weight[0].shape)

	def GenerateWeight(self, input_size, output_size):
		return np.random.rand(output_size, input_size+1)-0.5 # numer of theta between layer l and l+1 s(l+1)x(sl+1)
	# load pretrained weight to keep training
	def LoadWeight(self,list_weight):
		self.Weight = list_weight


	# activation function
	def Sigmoid(self,z):
		a = 1/(1+np.exp(-z))
		return a

	#Loss function
	def Cost(self,yhat,y):
		#MSE
		return np.mean(1/2*np.power(yhat - y,2))

	# Add Bias
	def AddBias(self,a): # a is input m x ln
		m,_ = a.shape
		bias = np.ones((m,1)) 
		return np.hstack([bias,a])
	
	#forward
	def Forward(self,X):
		list_a = []        #list to contain the value of matrix a
		list_a_bias = []   #list to contain the value of matrix a after add bias

		a1= X.copy()
		a1_bias = self.AddBias(a1)
		list_a.append(a1)
		list_a_bias.append(a1_bias)
		for i in range(len(self.Weight)): #i = l-1
			theta = self.Weight[i].copy()
			z = np.matmul(list_a_bias[i],theta.T)# (m,sl)
			a = self.Sigmoid(z)
			a_bias = self.AddBias(a)# (m,sl+1)
			list_a.append(a)
			list_a_bias.append(a_bias)
		return list_a, list_a_bias

	# backpropagation
	def backpropagation(self,list_a,list_a_bias):
		dW = [] 
		list_delta = []
		#loss derivative function can be simplify as (t-y)
		deltaLoss = list_a[self.L-1] - self.y
		Weight = self.Weight.copy()
		list_delta.append(deltaLoss)

		# calculate deltaW  to update, deltaw have form of matrix (ln+1,l(n+1))
		for l in range(self.L - 2, -1, -1):
			# calculate deltaW
			#deltaWi have the same shape as Wi
			DeltaJ = np.matmul(list_delta[0].T,list_a_bias[l]) # (m, l(n+1)).T x (m, ln+1)) = l(n+1),ln+1 
			# using chain rule, can be seen that the lower level layer DeltaJ will be update rely on the result by 
			# multiply the previous delta with the higher layer weight with activation sigmoid function da/dz = a*(1-a) 
			d_act = list_a[l]*(1-list_a[l]) #form (m,l(n+1))
			nextdelta = np.matmul(list_delta[0],Weight[l][:,1:]) * d_act  # [(m,l(n+1)) x (l+1, l(n+1))]*(m,ln)
			list_delta = [nextdelta] + list_delta 
			dW = [DeltaJ] + dW
			preds = list_a[-1]
		return preds,dW

	
	def re_Assigned(self,dW,lr):  #update new value for weight
		for i in range(len(self.Weight)):
			self.Weight[i] = self.Weight[i] - lr*dW[i]

	def train(self,epochs,learningrate):
		acc = []
		losses = []
		LOSS = []
		freq = 10 # print accuary and loss after freq epochs

		for epoch in range(epochs):
			list_a, list_a_bias = self.Forward(self.X) # forward
			preds,dW = self.backpropagation(list_a,list_a_bias)  # backward to get dL/dW
			self.re_Assigned(dW,learningrate)         # update new Weight
			#Calculate loss	
			loss = self.Cost(preds,self.y)
			losses.append(loss)
			#Calculate accuracy
			correct_case = 0
			for i,pred in enumerate(preds):
				if pred.argmax() == self.y[i].argmax(): # if the index of max(pred) = max(y) so the prediction is right
					correct_case += 1 
			cur_acc = (correct_case/self.m)*100 # calculate the accurary in percent
			acc.append(cur_acc)
			LOSS.append(np.array(losses[-3:]).mean())
			# print out the progress
			if (epoch+1) % freq == 0:
				avg_acc = np.array(acc[-3:]).mean()
				_loss = LOSS[-1]
				print(f"Epoch {epoch + 1}: AVG ACC during {freq} epochs: {avg_acc}")
				print(f"Epoch {epoch + 1}: Loss: {_loss}")
				print()
		return acc, LOSS
	def solfMax(self,y):
		y = np.power(np.e,y)
		return y/np.sum(y)
		
	def predict(self,X): # get the prediction 
		pred= self.Forward(X.reshape(1,self.n)) # a = y predicted
		return self.solfMax(pred)
	
	def visualize(self,acc,losses): # graph the accuary and losses to epochs
		fig, ax = plt.subplots(2)
		m = len(acc)
		epochs = np.arange(len(acc)).reshape(m,)
		acc = np.array(acc).reshape(m,)
		losses = np.array(losses).reshape(m,)
		fig.suptitle('Graph Loss and Accuary - epochs')
		ax[0].plot(epochs,acc)
		ax[0].set_title('Accuray-epochs')
		ax[0].set_xlabel("epochs")
		ax[0].set_ylabel("percent")
		ax[1].plot(epochs,losses,color ='red')
		ax[1].set_title('Loss-epochs')
		ax[1].set_xlabel("epochs")
		ax[1].set_ylabel("loss")
		plt.show()


	def Testing(self, Xtest,Ytest): # For estimate the performance of NN on not learnt dataset
		print("Testing...")
		m,_=Ytest.shape
		list_a,_ = self.Forward(Xtest)
		preds = list_a[-1]
		#Calculate accuracy
		correct_case = 0
		for i,pred in enumerate(preds):
			if pred.argmax() == Ytest[i].argmax():
				correct_case += 1 
		cur_acc = (correct_case/m)*100  
		loss = self.Cost(preds,Ytest)
		print("accuracy after test: ",cur_acc)
		print("total loss of Testing: ",loss)

