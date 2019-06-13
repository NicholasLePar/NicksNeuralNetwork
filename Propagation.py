import numpy as np

from Activation import activation_forward, activation_backward

from Normalization import z_score_normalization

from Regularization import forward_propagation_droput, backward_propagation_dropout, backward_propagation_l2_regularization
########################################################################################################################
"""
GROUP: Forward Propagation
	-The forward propdgation process, computes the output of the neural network
	-TODO: Work on Feature List

EXTERNAL FUNCTIONS: 
					1)forward_propogation: starts the forward propogation of the neural network
						-FEATURE:network architecture selection, right now just starts L_model_forward which is just
						      all ReLU neurons for hidden layers with the output layer being sigmoid neurons
						-FEATURE:network connectivity along with network achritecture
					
INTERNAL FUNCTIONS:
					1)L_model_forward: goes through all layers and runs forward propagation on them in the style below:
						-All hidden layer neurons are ReLU
						-Output layer neurons are Sigmoid
						-all densely connected
					2)linear_activation_forward:based on the activation type selected, computes the linear & activation 
						of all neurons in the layer.
						-FEATURE: unique activation selection for neurons within layer
						-FEATURE: connectivity? (where exactly should this happen)
					3)linear_forward: computes the linear component of a single layers forward propogation
						-FEATURE:connectivity

"""
########################################################################################################################

###EXTERNAL FUNCTIONS###

def forward_propagation(X, parameters, keep_prob):
	"""
	Implement forward propagation

	Arguments:
	X -- data, numpy array of shape (input size, number of examples)
	parameters -- output of initialize_parameters_deep()
	keep_prob - probability of keeping a neuron active during drop-out, scalar


	Returns:
	AL -- last post-activation value
    caches -- list of caches containing:
				every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
	"""
	
	AL,caches = L_model_forward(X, parameters, keep_prob)
	
	return AL, caches

###INTERNAL FUNCTIONS###

def L_model_forward(X, parameters, keep_prob):
	"""
	Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
	
	Arguments:
	X -- data, numpy array of shape (input size, number of examples)
	parameters -- output of initialize_parameters_deep()
	keep_prob - probability of keeping a neuron active during drop-out, scalar

	Returns:
	AL -- last post-activation value
	caches -- list of caches containing:
	            every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
	"""

	caches = []
	A = X
	L = len(parameters) // 2                  # number of layers in the neural network

	#A0 = inputs as will be needed to cover interation through backprop
	caches.append({"A0":X})

	# Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
	for l in range(1, L):
		A_prev = A 
		A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu", keep_prob, l, L)
		caches.append(cache)

	# Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
	AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid", keep_prob, L, L)

	caches.append(cache)



	return AL, caches

def linear_activation_forward(A_prev, W, b, activation_type, keep_prob, l, number_of_layers):
	"""
	Implement the forward propagation for the LINEAR->ACTIVATION layer

	Arguments:
	A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
	W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
	b -- bias vector, numpy array of shape (size of the current layer, 1)
	activation_type -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
	keep_prob - probability of keeping a neuron active during drop-out, scalar

	Returns:
	A -- the output of the activation function, also called the post-activation value 
	cache -- a python dictionary containing "linear_cache" and "activation_cache";
			stored for computing the backward pass efficiently
	"""

	#compute the linear forward part of forward propagation
	Z, linear_cache = linear_forward(A_prev,W,b)

	#compute the activation function based on the type selected for the layer
	A, activation_cache = activation_forward(activation_type,Z)
	
	#save the output of this layer as a cache for backward propagation of this layer
	cache={}
	cache["A"+str(l)] = A
	cache["W"+str(l)] = W
	cache["b"+str(l)] = b
	cache["Z"+str(l)] = Z

	#apply dropout regularization to the layer output A if keep_prob is less than 1
	#if keep_prob==1 that would signify all neurons are active.
	if keep_prob < 1 and l < number_of_layers:
		A, DL = forward_propagation_droput(A,keep_prob)
		cache["dropout"+str(l)]= DL

	return A, cache

def linear_forward(A, W, b):
	"""
	Implement the linear part of a layer's forward propagation.

	Arguments:
	A -- activations from previous layer (or input data): (size of previous layer, number of examples)
	W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
	b -- bias vector, numpy array of shape (size of the current layer, 1)

	Returns:
	Z -- the input of the activation function, also called pre-activation parameter 
	cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
	"""
	Z = np.dot(W,A)+b

	cache = (A, W, b)

	return Z, cache

########################################################################################################################
"""
GROUP: Backward Propagation
	-The backward propagation process, compute the gradients of each layers output
	
EXTERNAL FUNCTIONS: 
					1)backward_propagation: starts the backward propagation process of the neural network.
					-FEATURE:same as forward prop external function
						-FEATURE:network architecture selection, right now just starts L_model_forward which is just
						  all ReLU neurons for hidden layers with the output layer being sigmoid neurons
						-FEATURE:network connectivity along with network achritecture
INTERNAL FUNCTIONS:
					1)L_model_backward: Implements backpropagation through all the layers as if output layer was a
					sigmoid and the hiddern layers were ReLu
					2)linear_activation_backward: computes the backward propagtion of the activation & linear part of
					a layer.
						-FEATURE: same as linear_activation_forward, connectivity, selectivity of activation, etc.
					3)linear_backward: computes the linear part of backward propagtion for a single layer
						-FEATURE: same as linear_forward, connectivity, etc.
"""
########################################################################################################################

### EXTERNAL FUNCTIONS ###
def backward_propagation(AL, Y, caches, lambd, keep_prob):
	"""
	Implement backward propagation

	Arguments:
	AL -- probability vector, output of the forward propagation (L_model_forward())
	NOTE: Should be cost not Y?
	Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
	cost

	caches -- list of caches containing:
				every cache of linear_activation_forward() with "relu" (it's caches[l],
				for l in range(L-1) i.e l = 0...L-2)
				the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
	lambd

	Returns:
	grads -- A dictionary with the gradients
			grads["dA" + str(l)] = ... 
			grads["dW" + str(l)] = ...
			grads["db" + str(l)] = ... 
	"""
	grads = L_model_backward(AL, Y, caches, lambd, keep_prob)
	return grads

### INTERNAL FUNCTIONS ###

def L_model_backward(AL, Y, caches, lambd, keep_prob):
	"""
	Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

	Arguments:
	AL -- probability vector, output of the forward propagation (L_model_forward())

	NOTE: Should be cost not Y?
	Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
	cost

	caches -- list of caches containing:1
				every cache of linear_activation_forward() with "relu" (it's caches[l],
				for l in range(L-1) i.e l = 0...L-2)
				the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
	lambd

	Returns:
	grads -- A dictionary with the gradients
			grads["dA" + str(l)] = ... 
			grads["dW" + str(l)] = ...
			grads["db" + str(l)] = ... 
	"""
	grads = {}
	L = len(caches) - 1 # the number of layers, subtract 1 as A0=X is not counted as a layer of the NN
	m = AL.shape[1]
	Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL


	# Initializing the backpropagation
	#NOTE: this is the derivative of the cost with respect to AL (in this case AL is a sigmoid activation output)
	# this will need to be adjusted for other types
	dAL = -(np.divide(Y, AL, where=AL!=0) - np.divide(1 - Y, 1 - AL, where=AL!=1))
	dAL[dAL==-np.inf]=0

	# Lth layer (SIGMOID -> LINEAR) gradients. 
	#Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
	grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
																									caches,
																									'sigmoid',
																									lambd,
																									keep_prob,
																									L)

	# Loop from l=L-2 to l=0
	for l in reversed(range(L-1)):
		# lth layer: (RELU -> LINEAR) gradients.
		# Inputs: "grads["dA" + str(l + 1)], current_cache".
		# Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
		dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], 
																	caches,
																	'relu', 
																	lambd,
																	keep_prob,
																	l+1)
		grads["dA" + str(l)] = dA_prev_temp
		grads["dW" + str(l + 1)] = dW_temp
		grads["db" + str(l + 1)] = db_temp
	return grads

def linear_activation_backward(dA, caches, activation_type, lambd, keep_prob,l):
	"""
	Implement the backward propagation for the LINEAR->ACTIVATION layer.

	Arguments:
	dA -- post-activation gradient for current layer l 
	caches -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
	activation_type -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
	lambd

	Returns:
	dAL-- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	db -- Gradient of the cost with respect to b (current layer l), same shape as b
	"""
	#unpack the inner caches
	cache_current_layer = caches[l]
	cache_previous_layer = caches[l-1]

	#unpack variables to use from cache in layers
	A = cache_previous_layer["A"+str(l-1)] 
	W = cache_current_layer["W"+str(l)] 
	b = cache_current_layer["b"+str(l)] 
	Z = cache_current_layer["Z"+str(l)] 

	#perform backpropagation for activation part of the layer
	dZ = activation_backward(activation_type, dA, Z)


	#perform backpropagation for the linear part
	dAL, dW, db = linear_backward(dZ, (A,W,b), lambd)

	#apply dropout regularization to the layer during backprop if keep_prob is less than 1 
	#if keep_prob==1 that would signify all neurons are active.
	if keep_prob<1 and l>1:
		DL = cache_previous_layer["dropout"+str(l-1)]
		dAL = backward_propagation_dropout(dAL,DL,keep_prob)


	return dAL, dW, db

def linear_backward(dZ, cache, lambd):
	"""
	Implement the linear portion of backward propagation for a single layer (layer l)

	Arguments:
	dZ -- Gradient of the cost with respect to the linear output (of current layer l)
	cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
	lambd

	Returns:
	dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	dW -- Gradient of the cost with respect to W (current layer l), same shape as W
	db -- Gradient of the cost with respect to b (current layer l), same shape as b
	"""
	A_prev, W, b = cache
	m = A_prev.shape[1] #number of examples

	#compute the linear gradient for backpropagtion with no regularization term if lambd is zero
	if lambd ==0:
		dW = (1.0/m)*np.dot(dZ,A_prev.T)
	#otherwise compute the backward proprogation with the regularization
	else:
		dW = (1.0/m)*np.dot(dZ,A_prev.T) + backward_propagation_l2_regularization(m,W,lambd)

	#compute the bias gradients
	db = (1.0/m)*np.sum(dZ,axis=1,keepdims=True)

	#gradient of the cost function with respect to activation function
	dA_prev = np.dot(W.T,dZ)

	assert (dA_prev.shape == A_prev.shape)
	assert (dW.shape == W.shape)
	assert (db.shape == b.shape)

	return dA_prev, dW, db
