import numpy as np


def NumToVec(n, maxLength = 10):
	vec = np.zeros(maxLength)
	vec[n] = 1
	vec = vec.reshape(1,maxLength)
	return vec  #1,k

def convert_Y(Ys): # tạo ra 1 list các vector y Ví dụ:[0,1,0,0],[1,0,0,0]... 
		values = np.unique(Ys,axis=1)
		n,m=Ys.shape
		k = len(values[0])
		new_Y = []
		 #ví dụ với K=4 : [0,0,0,0]
		for i in range(m):
			value = Ys[0][i]
			x,n = np.where(values==value)
			Y= NumToVec(n,k) 
			new_Y.append(Y)
		new_Y = np.array(new_Y).reshape(m,k)
		return values, new_Y # (m,k))

#####################################################
class ShallowNeuralNet():
    def __init__(self, Xs, Ys, input_size, hidden_size, output_size):
        self.Xs = Xs #(2000, 784) List(nparray)
        self.Ys = Ys #(2000, 10) List(nparray)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # self.w1 = self.GenerateWeight(input_size, hidden_size) #(785, 81)
        # self.w2 = self.GenerateWeight(hidden_size, output_size) #(82, 10)

    def GenerateWeight(self, input_size, output_size):
        return np.random.rand(output_size, input_size+1).transpose() -0.5

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
    
    def train(self, epochs, w1, w2, lr):
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
                pred = self.Forward(x, w1, w2)
                if pred.argmax() == y.argmax():
                    correct_case+=1
                l.append(self.MSE(pred, y))
                delta1, delta2 = self.Backward(x, y, w1, w2)
                cap_delta1+= delta1
                cap_delta2+= delta2
            w1 = w1 - (lr*(cap_delta1))
            w2 = w2 - (lr*(cap_delta2))
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
        return acc, losses, w1, w2
# multi_layer_neural_network    
class NeuralNetWork():
	def __init__(self,Xs,Ys,K,lr,rl,list_of_hidden_layer):
		# self.Xs = self.normalize(Xs,True) # mxn
		self.Xs=Xs
		self.m,self.n = self.Xs.shape

		self.Ys = Ys
		self.K = K
		self.lr = lr
		self.rl = rl
		self.L = len(list_of_hidden_layer) + 2 # number of layer

		self.list_of_Node = [self.n ] + list_of_hidden_layer + [self.K]
		print("list of Node",self.list_of_Node)
		self.list_theta=[]
		

	def init_theta(self):
		for l in range(self.L-1):
			sample_theta = self.generate_random_weight(l,self.list_of_Node)
			print(f"sample theta {l+1}: {sample_theta.shape}")
			self.list_theta.append(sample_theta)

	def generate_random_weight(self,l,unit_in_layer):
		sample_theta=np.random.rand(unit_in_layer[l+1],unit_in_layer[l]+1)-0.5 # numer of theta between layer l and l+1 s(l+1)x(sl+1)
		return sample_theta

	def AddBias(self, input): # input (m,n)
		m,n =input.shape
		output = np.concatenate((np.ones([m,1]),input),axis = 1)
		return output # (m,n+1)


	def normalize(self,Xs,nor=True):
		m,n = Xs.shape
		if nor == True:
			self.mean_value = np.mean(Xs,axis=0).reshape(1,n)
			max_value = np.amax(Xs,axis=0).reshape(1,n)
			min_value =np.amin(Xs,axis=0).reshape(1,n)
			self.rangeMtx = max_value-min_value
		scaled_Xs=(Xs-self.mean_value)/self.rangeMtx

		return   scaled_Xs

	def sigmoid(self, z):
		return 1/(1+np.exp(-z))
	
	def hypothesis_sigmoid(self,a,cur_theta):
		z= np.matmul(a,cur_theta.T)  #((m,sl+1)x(sl+1,s(l+1)) = m,s(l+1)
		hx= self.sigmoid(z)
		return hx #m,s(l+1)
	
	def ForwardProp(self,Xs):
		a1=Xs # (m,n)
		a1_bias = self.AddBias(a1) #(m,n+1)
		list_a=[]
		list_a_bias=[]
		list_a.append(a1)
		list_a_bias.append(a1_bias)
		for i in range(len(self.list_theta)): #i = l-1
			theta = self.list_theta[i].copy()
			a = self.hypothesis_sigmoid(list_a_bias[-1],theta) # (m,sl)
			a_bias=self.AddBias(a) # (m,sl+1)
			list_a.append(a)
			list_a_bias.append(a_bias)
		return list_a, list_a_bias
	# len(lis_a)=self.L

	def BackwardProp(self):
		dJ=[]
		list_delta = []
		list_a,list_a_bias = self.ForwardProp(self.Xs)
		delta_L = list_a[self.L-1] - self.Ys # (m,k)
		list_delta.append(delta_L)
		list_theta=self.list_theta.copy()
		# len(list_theta) = self.L-1	
		for l in range(self.L-2,-1,-1):
			delta = np.matmul(list_delta[0],list_theta[l][:,1:])*list_a[l]*(1-list_a[l]) # [(m,s(l+1)) x (s(l+1),sl+1)]*(m,sl)
			Delta = np.matmul(list_delta[0].T,list_a_bias[l]) # (m, s(l+1)).T x (m, sl+1)
			list_delta = [delta] + list_delta 
			dJ = [Delta] + dJ
		pred= list_a[-1] # mxk
		return dJ,pred
	
	def MSE(self, pred, y):
		return np.mean(np.power(pred - y,2))

	def regularization_reduce(self, theta,n):		
		regularization_theta0=np.zeros([n,1]).reshape(n,1)      # (s(l+1),1)
		regularization_thetaj=(self.rl/self.m)*theta[:,1:]      # (s(l+1),sl)
		regularization_reduce= np.concatenate([regularization_theta0,regularization_thetaj],axis=1)
		return regularization_reduce

	def train(self,epochs):
		acc = []
		losses = []
		for epoch in range(epochs):
			# ReAssign
			theta_1 = self.list_theta[0]
			dJ, preds = self.BackwardProp()
			for i in range(len(self.list_theta)):
				theta = self.list_theta[i].copy()	
				dtheta = dJ[i]
				sl_1,sl = theta.shape
				theta =theta -self.lr*dtheta  # (s(l+1),sl+1)
				self.list_theta[i] = theta
			#Calculate loss	
			loss = self.MSE(preds,self.Ys)
			losses.append(loss)
			#Calculate accuracy
			correct_case = 0
			for i,pred in enumerate(preds):
				if pred.argmax() == self.Ys[i].argmax():
					correct_case += 1 
			cur_acc = (correct_case/self.m)*100
			acc.append(cur_acc)

			freq = 5
			if (epoch+1) % freq == 0:
				avg_acc = np.array(acc[-3:]).mean()
				_loss = np.array(losses[-3:]).mean()
				print(f"Epoch {epoch + 1}: AVG ACC during {freq} epochs: {avg_acc}")
				print(f"Epoch {epoch + 1}: Loss: {_loss}")
				print()
		return acc, losses
class PredictNeuralNetWork():
	def __init__(self,Xs_predict,list_weigth_trained):
		self.Xs_predict = Xs_predict
		self.list_theta =list_weigth_trained
		self.K,h = list_weigth_trained[-1].shape
		# print("K = ",self.K)
		# for sample_theta in list_weigth_trained:
		# 	print(sample_theta.shape)
	def AddBias(self, input): # input (m,n)
		m,n =input.shape
		output = np.concatenate((np.ones([m,1]),input),axis = 1)
		return output # (m,n+1)

	def sigmoid(self, z):
		return 1/(1+np.exp(-z))

	def hypothesis_sigmoid(self,a,cur_theta):
		z= np.matmul(a,cur_theta.T)  #((m,sl+1)x(sl+1,s(l+1)) = m,s(l+1)
		hx= self.sigmoid(z)
		return hx #m,s(l+1)

	def estimate(self):
		a1=self.Xs_predict # (m,n)
		a1_bias = self.AddBias(a1) #(m,n+1)
		list_a=[]
		list_a_bias=[]
		list_a.append(a1)
		list_a_bias.append(a1_bias)
		for i in range(len(self.list_theta)): #i = l-1
			theta = self.list_theta[i].copy()
			a = self.hypothesis_sigmoid(list_a_bias[-1],theta) # (m,sl)
			a_bias=self.AddBias(a) # (m,sl+1)
			list_a.append(a)
			list_a_bias.append(a_bias)
		pred =list_a[-1] # m,k
		m,n=pred.shape
		output = []
		for i in range(m):
			index = pred[i].argmax()
			Y= NumToVec(index,self.K) 
			output.append(Y)
		output = np.array(output).reshape(m,self.K)
		
		return output