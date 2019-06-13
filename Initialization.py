import numpy as np

########################################################################################################################
"""
GROUP: Parameter Initialization
	-Handles the initialization of the neural networks weights and biases 
	
EXTERNAL FUNCTIONS: 
					1) initialize_parameters: initializes the weights and bias according to the init_type selected
					
INTERNAL FUNCTIONS:
					1)initializ1e_parameters_xavier: uses the "Xavier" algorith for initialization
					2)initializ1e_parameters_he: uses the "He" algorithm for initialization
					3)initialize_parameters_random: initializes the weights by randNum[0,1)*scale and the biases to zero
"""
########################################################################################################################

###EXTERNAL FUNCTIONS###
def initialize_parameters(layers_dims, init_type, weight_scale=1, seed=3):
	"""
	Description:
	initializes the weights and bias according to the init_type selected.

	Arguments:
	layer_dims -- python array (list) containing the size of each layer.
	init_type -- the type of initialize method to use. 
	
	Optional Arguments:
	weight_scale -- the scale to which an initialization technique will assign weights
	seed -- seed use to intialize the numpy random function

	Returns:
	parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
					W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
					b1 -- bias vector of shape (layers_dims[1], 1)
					...
					WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
					bL -- bias vector of shape (layers_dims[L], 1)
	"""
	if init_type == "random":
		parameters = initialize_parameters_random(layers_dims,weight_scale,seed)
	elif init_type == "he":
		parameters = initialize_parameters_he(layers_dims,seed)
	elif init_type == "xaiver":
		parameters = initialize_parameters_xavier(layers_dims,seed)		
	else:
		print("ERROR: intitialize_parameters - no init_type was selected")
		print("init_type=" + init_type)
		sys.exit(1)
	
	return parameters

###INTERNAL FUNCTIONS###
def initialize_parameters_xavier(layers_dims,seed):
	"""
	Description:
	Xavier initialization uses a scaling factor for the weights of `sqrt(1./layers_dims[l-1])` 

	Arguments:
	layer_dims -- python array (list) containing the size of each layer.
	seed -- seed use to intialize the numpy random function

	Returns:
	parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
					W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
					b1 -- bias vector of shape (layers_dims[1], 1)
					...
					WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
					bL -- bias vector of shape (layers_dims[L], 1)
	"""

	np.random.seed(seed)
	parameters = {}
	L = len(layers_dims)

	for l in range(1, L):
		parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(1.0/layers_dims[l-1])
		parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
	    
	return parameters

def initialize_parameters_he(layers_dims,seed):
	"""
	Description:
	He initialization is a published technique in 2015 similiar to Xavier initialization.
		-Xavier initialization uses a scaling factor for the weights of `sqrt(1./layers_dims[l-1])` 
		-He initialization would use `sqrt(2./layers_dims[l-1])
	He initialization recommended for layers with a ReLU activation. 


	Arguments:
	layer_dims -- python array (list) containing the size of each layer.
	seed -- seed use to intialize the numpy random function

	Returns:
	parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
					W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
					b1 -- bias vector of shape (layers_dims[1], 1)
					...
					WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
					bL -- bias vector of shape (layers_dims[L], 1)
	"""

	np.random.seed(seed)
	parameters = {}
	L = len(layers_dims)

	for l in range(1, L):
		parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2.0/layers_dims[l-1])
		parameters['b' + str(l)] = np.zeros((layers_dims[l],1))
	    
	return parameters

def initialize_parameters_random(layers_dims,weight_scale,seed):
	"""
	Description:
	initializes the weights of all neurons randomly between [0,1)*scale and their biases to zero.

	Arguments:
	layer_dims -- python array (list) containing the size of each layer.
	weight_scale -- scalar to adjust the weight of the random numbers
	seed -- seed use to intialize the numpy random function
	
	Returns:
	parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
					W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
					b1 -- bias vector of shape (layers_dims[1], 1)
					...
					WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
					bL -- bias vector of shape (layers_dims[L], 1)
	"""

	np.random.seed(seed)               # This seed makes sure your "random" numbers will be the as ours
	parameters = {}
	L = len(layers_dims)            # integer representing the number of layers

	for l in range(1, L):
		parameters['W' + str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*weight_scale
		parameters['b' + str(l)] = np.zeros((layers_dims[l],1))

	return parameters

