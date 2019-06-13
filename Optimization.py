import numpy as np

import math

########################################################################################################################
"""
GROUP: Optimization Initialization and Parameter Updating
	-Handles the Optimization selection and updating of the Neural Networks Weights and Biases
	
EXTERNAL FUNCTIONS: 
					1) initialize_optimizer: sets the desired optimizer and initializes any corresponding parameters for
						the selected optimizer.
					2)update_parameters: Updates the neural networks parameters (weights, biases) based on 
						the optomizer selected
					
INTERNAL FUNCTIONS:
					1)initialize_velocity: Initializes the velocity 
					2)initialize_adam: Initializes v (momentum) and s (RMSprop) for the ADAM optimizer 
					3)update_parameters_with_gd: Update parameters using one step of gradient descent
					4)update_parameters_with_momentum: Update parameters with Momentum
					5)update_parameters_with_adam:Update parameters using the ADAM optimizer, It combines ideas
						from RMSProp (described in lecture) and Momentum
"""
########################################################################################################################

###EXTERNAL FUNCTIONS###

def initialize_optimizer(parameters,optimizer_type):
	"""
	Description:
	sets the desired optimizer and initializes any corresponding parameters for the selected optimizer.

	Arguments:
	optimizer_type -- the type of optimizer to use for updating weights
	parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
				W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
				b1 -- bias vector of shape (layers_dims[1], 1)
				...
				WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
				bL -- bias vector of shape (layers_dims[L], 1)

	Returns:
	optimizer -- the optimizer information that tracks the optimizer and it's state.
	"""
	
	optimizer = {}

	#Standard Gradient Descent with no extra optimization
	if optimizer_type == "gd":
		optimizer["optimizer_type"] = optimizer_type

	#Momentum
	elif optimizer_type == "momentum":
		optimizer["optimizer_type"] = optimizer_type
		optimizer["v"] = initialize_velocity(parameters)
	
	#ADAM optimizer = Momentum + RMSProp + Bias Correction
	elif optimizer_type == "adam":
		optimizer["optimizer_type"] = optimizer_type
		optimizer["v"], optimizer["s"] = initialize_adam(parameters)
		optimizer["t"] = 0
	else:
		print("ERROR: intitialize_optimizer - no optimizer_type was selected")
		print("optimizer_type=" + optimizer_type)
		sys.exit(1)
	
	return optimizer

def update_parameters(parameters,grads,learning_rate,optimizer,beta1=0.9,beta2=0.999, epsilon=1e-8):
	"""
	Description:
	Updates the neural networks parameters (weights, biases) based on the optomizer selected

	Arguments:
	parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
				W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
				b1 -- bias vector of shape (layers_dims[1], 1)
				...
				WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
				bL -- bias vector of shape (layers_dims[L], 1)
	grads -- python dictionary containing your gradients to update each parameters:
					grads['dW' + str(l)] = dWl
					grads['db' + str(l)] = dbl
	learning_rate -- the learning rate, scalar.
	optimizer -- the optimizer information that tracks the optimizer and it's state.


	Optional Arguments:
	beta1 -- Exponential decay hyperparameter for the first moment estimates 
			-Used in: Momentum, ADAM
			-Common values for beta1 range from 0.8 to 0.999. If you don't feel inclined to tune this, beta = 0.9
				is often a reasonable default. 
	beta2 -- Exponential decay hyperparameter for the second moment estimates
				-Used in: ADAM(RMS PROP)
	epsilon -- hyperparameter preventing division by zero in Adam updates
				-Used in ADAM(RMS PROP)

	parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
				W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
				b1 -- bias vector of shape (layers_dims[1], 1)
				...
				WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
				bL -- bias vector of shape (layers_dims[L], 1)
	optimizer -- the optimizer information that tracks the optimizer and it's state.
	"""
	# Update parameters via GD
	if optimizer["optimizer_type"] == "gd":
		parameters = update_parameters_with_gd(parameters, grads, learning_rate)

	# Update pramaeters with Momentum
	elif optimizer["optimizer_type"] == "momentum":
		parameters, optimizer["v"] = update_parameters_with_momentum(parameters, grads, optimizer["v"], beta1,
									 learning_rate)
	#update parameters with ADAM
	elif optimizer["optimizer_type"] == "adam":
		optimizer["t"] = optimizer["t"] + 1 # Adam counter for bias correction
		parameters, optimizer["v"], optimizer["s"] = update_parameters_with_adam(parameters, grads, optimizer["v"], 
														optimizer["s"], optimizer["t"], learning_rate, beta1, beta2, 
														epsilon)
	else:
		print("ERROR: update_parameters - no optimizer_type was selected")
		print("optimizer_type=" + optimizer["optimizer_type"])
		sys.exit(1)

	return parameters, optimizer

###INTERNAL FUNCTIONS###

def initialize_velocity(parameters):
	"""
	Description:
	Initializes the velocity as a python dictionary with:
				- keys: "dW1", "db1", ..., "dWL", "dbL" 
				- values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
	Arguments:
	parameters -- python dictionary containing your parameters.
					parameters['W' + str(l)] = Wl
					parameters['b' + str(l)] = bl

	Returns:
	v -- python dictionary containing the current velocity.
					v['dW' + str(l)] = velocity of dWl
					v['db' + str(l)] = velocity of dbl
	"""

	L = len(parameters) // 2 # number of layers in the neural networks
	v = {}

	# Initialize velocity
	for l in range(L):
		v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
		v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)

	return v

def initialize_adam(parameters):
	"""
	Initializes v and s as two python dictionaries with:
				- keys: "dW1", "db1", ..., "dWL", "dbL" 
				- values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.

	Arguments:
	parameters -- python dictionary containing your parameters.
					parameters["W" + str(l)] = Wl
					parameters["b" + str(l)] = bl

	Returns: 
	v -- python dictionary that will contain the exponentially weighted average of the gradient.
					v["dW" + str(l)] = ...
					v["db" + str(l)] = ...
	s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
					s["dW" + str(l)] = ...
					s["db" + str(l)] = ...

	"""

	L = len(parameters) // 2 # number of layers in the neural networks
	v = {}
	s = {}

	# Initialize v, s. Input: "parameters". Outputs: "v, s".
	for l in range(L):
		v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
		v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
		s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
		s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)

	return v, s

def update_parameters_with_gd(parameters, grads, learning_rate):
	"""
	-Update parameters using one step of gradient descent
	-A simple optimization method in machine learning is gradient descent (GD). When you take gradient steps with 
		respect to all m examples on each step, it is also called Batch Gradient Descent. subset of M is minibatch GD
		and a single m example is Stochastic Gradient Descent

	Arguments:
	parameters -- python dictionary containing your parameters to be updated:
					parameters['W' + str(l)] = Wl
					parameters['b' + str(l)] = bl
	grads -- python dictionary containing your gradients to update each parameters:
					grads['dW' + str(l)] = dWl
					grads['db' + str(l)] = dbl
	learning_rate -- the learning rate, scalar.

	Returns:
	parameters -- python dictionary containing your updated parameters 
	"""

	L = len(parameters) // 2 # number of layers in the neural networks

	# Update rule for each parameter
	for l in range(L):
		parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
		parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

	return parameters

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
	"""
	Update parameters using Momentum
	-Because mini-batch gradient descent makes a parameter update after seeing just a subset of examples,
		the direction of the update has some variance, and so the path taken by mini-batch gradient descent will 
		"oscillate" toward convergence. Using momentum can reduce these oscillations. 
	-Momentum takes into account the past gradients to smooth out the update. We will store the 'direction' of the 
		previous gradients in the variable v. Formally, this will be the exponentially weighted average of the gradient on
		previous steps. You can also think of v as the "velocity" of a ball rolling downhill, building up speed 
		(and momentum) according to the direction of the gradient/slope of the hill.
	-The velocity is initialized with zeros. So the algorithm will take a few iterations to "build up" 
		velocity and start to take bigger steps.
	-If beta = 0, then this just becomes standard gradient descent without momentum.
	-The larger the momentum beta is, the smoother the update because the more we take the past gradients into account. 
		But if beta is too big, it could also smooth out the updates too much. 

	Arguments:
	parameters -- python dictionary containing your parameters:
					parameters['W' + str(l)] = Wl
					parameters['b' + str(l)] = bl
	grads -- python dictionary containing your gradients for each parameters:
					grads['dW' + str(l)] = dWl
					grads['db' + str(l)] = dbl
	v -- python dictionary containing the current velocity:
					v['dW' + str(l)] = ...
					v['db' + str(l)] = ...
	beta -- the momentum hyperparameter, scalar
	learning_rate -- the learning rate, scalar

	Returns:
	parameters -- python dictionary containing your updated parameters 
	v -- python dictionary containing your updated velocities
	"""

	L = len(parameters) // 2 # number of layers in the neural networks

	# Momentum update for each parameter
	for l in range(L):        
		# compute velocities
		v["dW" + str(l+1)] = beta*v["dW" + str(l+1)] + (1-beta)*grads['dW' + str(l+1)]
		v["db" + str(l+1)] = beta*v["db" + str(l+1)] + (1-beta)*grads['db' + str(l+1)]
		# update parameters with momentum
		parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*v["dW" + str(l+1)] 
		parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*v["db" + str(l+1)] 

	return parameters, v

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon):
	"""
	Update parameters using Adam
	-Adam is one of the most effective optimization algorithms for training neural networks. 
		It combines ideas from RMSProp (described in lecture) and Momentum.
	-RMSprop  intuition is that if B(bias) is vertical and W(Weight) is horizontal on a bowl topography,
		and we want to go fast horizontal and slow veritcal.
	1.It calculates an exponentially weighted average of past gradients, 
		and stores it in variables v (before bias correction) and v-corrected (with bias correction). 
	2.It calculates an exponentially weighted average of the squares of the past gradients, 
		and stores it in variables s(before bias correction) and s-corrected (with bias correction). 
	3.It updates parameters in a direction based on combining information from "1" and "2".

	Arguments:
	parameters -- python dictionary containing your parameters:
					parameters['W' + str(l)] = Wl
					parameters['b' + str(l)] = bl
	grads -- python dictionary containing your gradients for each parameters:
					grads['dW' + str(l)] = dWl
					grads['db' + str(l)] = dbl
	v -- Adam variable, moving average of the first gradient, python dictionary
	s -- Adam variable, moving average of the squared gradient, python dictionary
	t -- counts the number of steps taken of Adam used in bias correction
	learning_rate -- the learning rate, scalar.
	beta1 -- Exponential decay hyperparameter for the first moment estimates 
	beta2 -- Exponential decay hyperparameter for the second moment estimates 
	epsilon -- hyperparameter preventing division by zero in Adam updates

	Returns:
	parameters -- python dictionary containing your updated parameters 
	v -- Adam variable, moving average of the first gradient, python dictionary
	s -- Adam variable, moving average of the squared gradient, python dictionary
	"""

	L = len(parameters) // 2                 # number of layers in the neural networks
	v_corrected = {}                         # Initializing first moment estimate, python dictionary
	s_corrected = {}                         # Initializing second moment estimate, python dictionary

	# Perform Adam update on all parameters
	for l in range(L):
		# Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
		v["dW" + str(l+1)] = beta1*v["dW" + str(l+1)] + (1.0-beta1)*grads['dW' + str(l+1)]
		v["db" + str(l+1)] = beta1*v["db" + str(l+1)] + (1-beta1)*grads['db' + str(l+1)]

		# Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
		v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)]/(1.0-(math.pow(beta1,t)))
		v_corrected["db" + str(l+1)] = v["db" + str(l+1)]/(1.0-(math.pow(beta1,t)))

		# Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
		s["dW" + str(l+1)] = beta2*s["dW" + str(l+1)] + (1.0-beta2)*(np.power(grads['dW' + str(l+1)],2.0))
		s["db" + str(l+1)] = beta2*s["db" + str(l+1)] + (1.0-beta2)*(np.power(grads['db' + str(l+1)],2.0))

		# Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
		s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)]*(1.0-(math.pow(beta2,t)))
		s_corrected["db" + str(l+1)] = s["db" + str(l+1)]*(1.0-(math.pow(beta2,t)))

		# Update parameters with ADAM optimizer. 
		#Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
		parameters["W" + str(l+1)] = (
										parameters["W" + str(l+1)] 
										- learning_rate 
										* ( v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)])
										+ epsilon))
									)
		parameters["b" + str(l+1)] = (
										parameters["b" + str(l+1)] 
										- learning_rate 
										* ( v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) 
										+ epsilon))
									)
	
	return parameters, v, s
