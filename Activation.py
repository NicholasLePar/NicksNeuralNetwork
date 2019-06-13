import numpy as np

import sys
########################################################################################################################
"""
GROUP: Activation Functions
	-Collection of Activation Function and there forward/backward propagation equation
	-TODO: implement more activations ie softmax
	
EXTERNAL FUNCTIONS: 
					1) activation_forward: selects the activation to run for forward propogation and returns its
						activation and cache.
					2) activation_backward: selects the activation to run for backward propogation and returns its
						activation and cache.
					
INTERNAL FUNCTIONS:
					1)sigmoid_forward
					2)sigmoid_backward
					3)relu_forward
					4)relu_backward
"""
########################################################################################################################

### EXTERNAL FUNCTIONS ###

def activation_forward(activation_type,Z):
	"""
	Implements the forward activation depending on the activation type specified

	Arguments:
	activation_type - activivation function to apply to the output of Z
	Z -- numpy array of any shape

	Returns:
	A -- output of activation, same shape as Z
	cache -- returns Z as well, useful during backpropagation
	"""
	if activation_type == "relu":
		A, cache = relu_forward(Z)
	elif activation_type == "sigmoid":
		A, cache = sigmoid_forward(Z)
	else:
		print('ERROR: activation_forward - no activation was selected')
		print("activation_type=" + activation_type)
		sys.exit(1)
	
	return A, cache

def activation_backward(activation_type, dA, cache):
	"""
	Implement the backward propagation for a single activation unit.

	Arguments:
	dA -- post-activation gradient, of any shape
	cache -- 'Z' where we store for computing backward propagation efficiently
	activation_type - activivation function to apply to the output of Z

	Returns:
	dZ -- Gradient of the cost with respect to Z
	"""
	if activation_type == "relu":
		dZ = relu_backward(dA, cache)
	elif activation_type == "sigmoid":
		dZ = sigmoid_backward(dA, cache)
	else:
		print("ERROR: activation_backward - no activation was selected")
		print("activation_type=" + activation_type)
		sys.exit(1)

	return dZ

### INTERNAL FUNCTIONS ###

def sigmoid_forward(Z):
	"""
	Implements the sigmoid activation in numpy

	Arguments:
	Z -- numpy array of any shape

	Returns:
	A -- output of sigmoid(z), same shape as Z
	cache -- returns Z as well, useful during backpropagation
	"""
	try:
		Z = np.clip(Z,-20,20)
		A = 1.0/(1.0+np.exp(-Z))
		cache = Z
	except RuntimeWarning as w:
		print("***sigmoid_forward***")
		print("w:")
		print(w)
		print("Z:")
		print(Z.shape)
		print(Z)
		sys.exit()

	return A, cache

def sigmoid_backward(dA, cache):
	"""
	Implement the backward propagation for a single SIGMOID unit.

	Arguments:
	dA -- post-activation gradient, of any shape
	cache -- 'Z' where we store for computing backward propagation efficiently

	Returns:
	dZ -- Gradient of the cost with respect to Z
	"""

	Z = cache
	try:
		Z = np.clip(Z,-20,20)
		s = 1.0/(1.0+np.exp(-Z))
		dZ = dA * s * (1.0 - s)
	except RuntimeWarning as w:
		print("***sigmoid_backward***")
		print(w)
		print(Z)

	return dZ

def relu_forward(Z):
	"""
	Implement the RELU function.

	Arguments:
	Z -- Output of the linear layer, of any shape

	Returns:
	A -- Post-activation parameter, of the same shape as Z
	cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
	"""

	A = np.maximum(0,Z)

	cache = Z

	return A, cache

def relu_backward(dA, cache):
	"""
	Implement the backward propagation for a single RELU unit.

	Arguments:
	dA -- post-activation gradient, of any shape
	cache -- 'Z' where we store for computing backward propagation efficiently

	Returns:
	dZ -- Gradient of the cost with respect to Z
	"""

	Z = cache
	dZ = np.array(dA, copy=True) # just converting dz to a correct object.

	# When z <= 0, you should set dz to 0 as well. 

	#print(dZ.shape)
	#print(dZ)
	#print(Z.shape)
	#print(Z)

	assert (dZ.shape == Z.shape)
	
	dZ[Z <= 0] = 0



	return dZ
