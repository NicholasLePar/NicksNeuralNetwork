from misc import *

from Normalization import z_score_normalization

from NicksNeuralNetwork import model
########################################################################################################################
"""
GROUP: Test cases

EXTERNAL FUNCTIONS: 
					1)test_sweep - runs through all the test cases covered in deeplearning.ai
					
INTERNAL FUNCTIONS:
					optimization_gd_test_case -standard gradient descent
					optimization_momentum_test_case - gradient descent with momentum
					optimization_adam_test_case - ADAM optimization 
					regularization_none_test_case - gradient descent with no regularization
					regularization_l2_test_case - gradient descent with l2 regularization
					regularization_dropout_test_case - gradient descent with dropout regularization

"""
########################################################################################################################
def test_sweep():
	print("Running sweep of all tests in Test.py")

	print("")
	print("********************************************************************************")
	print("*")
	print("* optimization_gd_test_case")
	print("*")
	print("********************************************************************************")
	print("")
	optimization_gd_test_case()

	print("")
	print("********************************************************************************")
	print("*")
	print("* optimization_momentum_test_case")
	print("*")
	print("********************************************************************************")
	print("")
	optimization_momentum_test_case()

	print("")
	print("********************************************************************************")
	print("*")
	print("* optimization_adam_test_case")
	print("*")
	print("********************************************************************************")
	print("")
	optimization_adam_test_case()

	print("")
	print("********************************************************************************")
	print("*")
	print("* regularization_none_test_case")
	print("*")
	print("********************************************************************************")
	print("")
	regularization_none_test_case()

	print("")
	print("********************************************************************************")
	print("*")
	print("* regularization_l2_test_case")
	print("*")
	print("********************************************************************************")
	print("")
	regularization_l2_test_case()

	print("")
	print("********************************************************************************")
	print("*")
	print("* regularization_dropout_test_case")
	print("*")
	print("********************************************************************************")
	print("")
	regularization_dropout_test_case()

def optimization_gd_test_case():

	train_X, train_Y = load_dataset()
	
	X = train_X
	Y = train_Y

	layers_dims = [X.shape[0], 5, 2, 1]
	init_type = "he"
	optimizer_type = "gd"
	learning_rate = 0.0007
	lambd = 0.0
	keep_prob = 1.0
	mini_batch_size = 64
	beta1 = 0.9
	beta2 = 0.999
	epsilon = 1e-8
	num_epochs = 10000
	print_cost = True

	parameters = model(X, Y, layers_dims, init_type, optimizer_type, learning_rate, lambd, keep_prob, mini_batch_size, beta1, beta2,
		epsilon, num_epochs, print_cost)

	# Predict
	predictions = predict(X, Y, parameters)

	# Plot decision boundary
	plt.title("Model")
	axes = plt.gca()
	axes.set_xlim([-1.5,2.5])
	axes.set_ylim([-1,1.5])
	plot_decision_boundary(lambda x: predict_dec(parameters, x.T), X, Y)

def optimization_momentum_test_case():
	print("NicksNeuralNetwork")

	train_X, train_Y = load_dataset()
	
	X = train_X
	Y = train_Y

	layers_dims = [X.shape[0], 5, 2, 1]
	init_type = "he"
	optimizer_type = "momentum"
	learning_rate = 0.0007
	lambd = 0.0
	keep_prob = 1.0
	mini_batch_size = 64
	beta1 = 0.9
	beta2 = 0.999
	epsilon = 1e-8
	num_epochs = 10000
	print_cost = True

	parameters = model(X, Y, layers_dims, init_type, optimizer_type, learning_rate, lambd, keep_prob, mini_batch_size, beta1, beta2,
		epsilon, num_epochs, print_cost)

	# Predict
	predictions = predict(X, Y, parameters)

	# Plot decision boundary
	plt.title("Model")
	axes = plt.gca()
	axes.set_xlim([-1.5,2.5])
	axes.set_ylim([-1,1.5])
	plot_decision_boundary(lambda x: predict_dec(parameters, x.T), X, Y)

def optimization_adam_test_case():
	print("NicksNeuralNetwork")

	train_X, train_Y= load_dataset()
	
	X = train_X
	Y = train_Y

	layers_dims = [X.shape[0], 5, 2, 1]
	init_type = "he"
	optimizer_type = "adam"
	learning_rate = 0.0007
	lambd = 0.0
	keep_prob = 1.0
	mini_batch_size = 64
	beta1 = 0.9
	beta2 = 0.999
	epsilon = 1e-8
	num_epochs = 10000
	print_cost = True

	#create trained neural network parameters
	parameters = model(X, Y, layers_dims, init_type, optimizer_type, learning_rate, lambd, keep_prob, mini_batch_size, beta1, beta2,
		epsilon, num_epochs, print_cost)

	# Predict
	predictions = predict(X, Y, parameters)

	# Plot decision boundary
	plt.title("Model")
	axes = plt.gca()
	axes.set_xlim([-1.5,2.5])
	axes.set_ylim([-1,1.5])
	plot_decision_boundary(lambda x: predict_dec(parameters, x.T), X, Y)

def regularization_none_test_case():
	print("NicksNeuralNetwork")

	train_X, train_Y, test_X, test_Y  = load_2D_dataset()
	
	X = train_X
	Y = train_Y

	layers_dims = [X.shape[0], 20, 3, 1]
	init_type = "xaiver"
	optimizer_type = "gd"
	learning_rate = 0.3
	lambd = 0.0
	keep_prob = 1.0
	mini_batch_size = X.shape[1]
	beta1 = 0.9
	beta2 = 0.999
	epsilon = 1e-8
	num_epochs = 30000
	print_cost = True

	#create trained neural network parameters
	parameters = model(X, Y, layers_dims, init_type, optimizer_type, learning_rate, lambd, keep_prob, mini_batch_size, beta1, beta2,
		epsilon, num_epochs, print_cost)

	# Predict
	print ("On the training set:")
	predictions_train = predict(train_X, train_Y, parameters)
	print ("On the test set:")
	predictions_test = predict(test_X, test_Y, parameters)

	# Plot decision boundary
	plt.title("Model without regularization")
	axes = plt.gca()
	axes.set_xlim([-0.75,0.40])
	axes.set_ylim([-0.75,0.65])
	plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

def regularization_l2_test_case():
	print("NicksNeuralNetwork")

	train_X, train_Y, test_X, test_Y  = load_2D_dataset()
	
	X = train_X
	Y = train_Y

	layers_dims = [X.shape[0], 20, 3, 1]
	init_type = "xaiver"
	optimizer_type = "gd"
	learning_rate = 0.3
	lambd = 0.7
	keep_prob = 1.0
	mini_batch_size = X.shape[1]
	beta1 = 0.9
	beta2 = 0.999
	epsilon = 1e-8
	num_epochs = 30000
	print_cost = True

	#create trained neural network parameters
	parameters = model(X, Y, layers_dims, init_type, optimizer_type, learning_rate, lambd, keep_prob, mini_batch_size, beta1, beta2,
		epsilon, num_epochs, print_cost)

	# Predict
	print ("On the training set:")
	predictions_train = predict(train_X, train_Y, parameters)
	print ("On the test set:")
	predictions_test = predict(test_X, test_Y, parameters)

	# Plot decision boundary
	plt.title("Model without regularization")
	axes = plt.gca()
	axes.set_xlim([-0.75,0.40])
	axes.set_ylim([-0.75,0.65])
	plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

def regularization_dropout_test_case():
	print("NicksNeuralNetwork")

	train_X, train_Y, test_X, test_Y  = load_2D_dataset()

	X = train_X
	Y = train_Y

	layers_dims = [X.shape[0], 20, 3, 1]
	init_type = "xaiver"
	optimizer_type = "gd"
	#learning rate cannot be 0.3? as in example. Exploding/Vanishing Gradient
	learning_rate = 0.03
	lambd = 0.0
	keep_prob = 0.86
	mini_batch_size = X.shape[1]
	beta1 = 0.9
	beta2 = 0.999
	epsilon = 1e-8
	num_epochs = 30000
	print_cost = True

	#create trained neural network parameters
	parameters = model(X, Y, layers_dims, init_type, optimizer_type, learning_rate, lambd, keep_prob, mini_batch_size, beta1, beta2,
		epsilon, num_epochs, print_cost)

	# Predict
	print ("On the training set:")
	predictions_train = predict(train_X, train_Y, parameters)
	print ("On the test set:")
	predictions_test = predict(test_X, test_Y, parameters)

	# Plot decision boundary
	plt.title("Model without regularization")
	axes = plt.gca()
	axes.set_xlim([-0.75,0.40])
	axes.set_ylim([-0.75,0.65])
	plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)