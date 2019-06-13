import numpy as np

from Regularization import cost_l2_regularization

########################################################################################################################
"""
GROUP: Cost Function
	-Computes the cost function of your neural networks output based on it's output and the corresponding label 
	
EXTERNAL FUNCTIONS: 
					1)compute_cost: selects the method in which to compute cost, whether to have regularization at cost
					  	-FEATURE: hoping to add a cost_type function in the future and experiment with different costs
INTERNAL FUNCTIONS:
					1)cross_entropy_cost- computes cross-entropy error
					2)reg_utils_cost - cost function from reg_utils of deeplearning.ai appears similiar to cross-entropy
"""
########################################################################################################################

### EXTERNAL FUNCTIONS ###

def compute_cost(AL, Y, parameters,lambd,cost_type="cross_entropy_cost"):
	"""
	Implement the cost function based on what is passed in by cost type

	Arguments:
	AL -- post-activation, output of forward propagation, of shape (output size, number of examples)
	Y -- "true" labels vector, of shape (output size, number of examples)
	lambd -- the value used to compute l2 regularization
	
	Optional Arguments:
	cost_type -- the selected cost function to use.

	Returns:
	cost -- computed cost(loss) of the output
	"""

	#select the cost type based on cost_type passed in
	if(cost_type=="reg_utils_cost"):
		cost = reg_utils_cost(AL, Y)
	elif(cost_type=="cross_entropy_cost"):
		cost = cross_entropy_cost(AL, Y)
	else:
		print('ERROR: compute_cost - no cost function was selected')
		print("activation_type=" + activation_type)
		sys.exit(1)

	#if lambd is greater than zero compute l2 regularization for the cost function
	if lambd > 0:
		cost = cost + cost_l2_regularization(Y, parameters, lambd)

	return cost       

### INTERNAL FUNCTIONS ###

def cross_entropy_cost(AL, Y):
	"""
	Implement the cross_entropy cost function

	Arguments:
	AL -- post-activation, output of forward propagation, of shape (output size, number of examples)
	Y -- "true" labels vector, of shape (output size, number of examples)

	Returns:
	cost -- cross-entropy cost
	"""
	m = Y.shape[1]
	# Compute loss from AL and Y.
 	try:
 		cost = -(1.0/m)*np.nansum(np.multiply(Y,np.log(AL)) + np.multiply((1-Y),np.log(1-AL+1e-8)))
 	except RuntimeWarning as w:
 		print("***cross_entropy_cost***")
 		print(w)
 		print(AL)
 		#print(sys.float_info.min)
 		print(Y)
	#cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
	return cost

def reg_utils_cost(a3, Y):
	"""
	Implement the cost function

	Arguments:
	a3 -- post-activation, output of forward propagation
	Y -- "true" labels vector, same shape as a3

	Returns:
	cost - value of the cost function
	"""

	m = Y.shape[1]

	logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
	logprobs[logprobs==np.inf]=0
	cost = 1./m * np.nansum(logprobs)

	return cost
