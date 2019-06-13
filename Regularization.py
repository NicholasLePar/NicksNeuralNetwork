import numpy as np

########################################################################################################################
"""
GROUP: Regularization
	-Performs regularization at necessary steps
	-At present I can only see this existing as helper functions for specific times in forward/cost/backpropagtion
	 so most will externally exist to other functions
	
EXTERNAL FUNCTIONS: 
					1)cost_l2_regularization - computes the l2 regularization term when computing cost
					2)backward_propagation_l2_regularization - computes the l2 regularization term for backpropagation
INTERNAL FUNCTIONS:
					
"""
########################################################################################################################

### EXTERNAL FUNCTIONS ###

def cost_l2_regularization(Y,parameters,lambd):
	"""
	computes the L2 regularization term of a cost function. See formula...

	Arguments:
	Y -- "true" labels vector, of shape (output size, number of examples)
	parameters -- python dictionary containing parameters of the model
	lamdb -- regularization term for scaling of "smoothing"
	

	Returns:
	L2_regularization_cost - value of the regularized loss function term for computing cost
	"""
	#Number of layers
	L = len(parameters) // 2 
	m = Y.shape[1]
	#compute the squared weights summation
	l2_weight_sum=0
	for l in range(L):
		l2_weight_sum = l2_weight_sum + np.sum(np.square(parameters["W" + str(l+1)])) 
	
	#compute L2_regularization cost
	L2_regularization_cost = (1.0/m)*(lambd/2.0)*(l2_weight_sum) 

	return L2_regularization_cost
 
def backward_propagation_l2_regularization(m,W,lambd):
	"""
	computes the l2 regularization term for backpropagation of a single layer

	Arguments:
	m -- number of examples
	W -- weights of the layer
	lamdb -- regularization term for scaling of "ssmoothing"

	Returns:
	L2_regularization_backpropagation - value of the regularized loss function term for backpropagation
	"""
	L2_regularization_backpropagation = ((lambd/m)*W)

	return L2_regularization_backpropagation

def forward_propagation_droput(AL,keep_prob):
	"""
	Implements dropout of a single layer in forward propagation
	
	Arguments:
	AL - the output of a layer in the network postactivation function, of size[neurons_in_layer,1]
	keep_prob - probability of keeping a neuron active during drop-out, scalar
	
	Returns:
	AL_dropout-- output of the forward propagation of each neuron in a layer with dropout applied
	DL-- represents the dropout matrix of what neurons are active (1) and non-active(0), should be added to cache
	"""
	# Step 1: initialize matrix DL = np.random.rand(..., ...)
	DL = np.random.rand(AL.shape[0],AL.shape[1])
	# Step 2: convert entries of DL to 0 or 1 (using keep_prob as the threshold)
	DL = (DL<keep_prob)
	#print(DL)                                         
	# Step 3: shut down some neurons of Al
	AL = np.multiply(AL,DL)
	# Step 4: scale the value of neurons that haven't been shut down
	AL = AL/keep_prob 
	
	return AL, DL

def backward_propagation_dropout(dAL,DL,keep_prob):
	"""
	Implements dropout of a single layer in backward propagation
	
	Arguments:
	dAL - the gradient of a layer in the backpropagation process, of size[neurons_in_layer,1]
	DL-- represents the dropout matrix of what neurons are active (1) and non-active(0) in the layer
	keep_prob - probability of keeping a neuron active during drop-out, scalar
	
	Returns:
	dAL_dropout-- gradient of the backward propagation of each neuron in a layer with dropout applied
	"""
	# Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
	dAL = np.multiply(dAL,DL)
	# Step 2: Scale the value of neurons that haven't been shut down
	dAL = dAL/keep_prob
	
	return dAL
