import numpy as np

import scipy.io

import matplotlib.pyplot as plt

import sklearn.datasets

from Propagation import forward_propagation

########################################################################################################################
"""
GROUP:Misc Function 
	-Supporting functions mainly for test sweeps for loading data and plotting results.
	#TODO: properly sort these functions into better groups with a roadmap.

EXTERNAL FUNCTIONS: 
					1)load_dataset - loads the make_moons dataset from sklearn for test cases in Test
					2)plot_decision_boundary - plots the decsion boundary for some test cases in Test
					3)load_2D_dataset - loads a custom dataset from deeplearning.ai used in test cases in Test
					4)predict_dec - generates predicitions from the nerual network given input data
					5)
					
INTERNAL FUNCTIONS:
					NONE
"""
########################################################################################################################

def load_dataset():
	np.random.seed(3)
	train_X, train_Y = sklearn.datasets.make_moons(n_samples=300, noise=.2) #300 #0.2 
	# Visualize the data
	plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
	plt.show()
	train_X = train_X.T
	train_Y = train_Y.reshape((1, train_Y.shape[0]))

	return train_X, train_Y

def plot_decision_boundary(model, X, y):
	# Set min and max values and give it some padding
	x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
	y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
	h = 0.01
	# Generate a grid of points with distance h between them
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	# Predict the function value for the whole grid
	Z = model(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	# Plot the contour and training examples
	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
	plt.ylabel('x2')
	plt.xlabel('x1')
	plt.scatter(X[0, :], X[1, :], c=y.reshape(y.shape[1],), cmap=plt.cm.Spectral)
	plt.show()

def load_2D_dataset():
	data = scipy.io.loadmat('datasets/data.mat')
	train_X = data['X'].T
	train_Y = data['y'].T
	test_X = data['Xval'].T
	test_Y = data['yval'].T

	plt.scatter(train_X[0, :], train_X[1, :], c=train_Y.reshape(train_Y.shape[1],), s=40, cmap=plt.cm.Spectral);
	plt.show()

	return train_X, train_Y, test_X, test_Y

def predict_dec(parameters, X):
	"""
	Used for plotting decision boundary.
	
	Arguments:
	parameters -- python dictionary containing your parameters 
	X -- input data of size (m, K)
	
	Returns
	predictions -- vector of predictions of our model (red: 0 / blue: 1)
	"""
	
	# Predict using forward propagation and a classification threshold of 0.5
	a3, cache = forward_propagation(X, parameters,1)
	predictions = (a3 > 0.5)
	return predictions

def predict(X, y, parameters):
	"""
	This function is used to predict the results of a  n-layer neural network.

	Arguments:
	X -- data set of examples you would like to label
	parameters -- parameters of the trained model

	Returns:
	p -- predictions for the given dataset X
	"""
	
	m = X.shape[1]
	p = np.zeros((1,m), dtype = np.int)
	
	# Forward propagation
	a3, caches = forward_propagation(X, parameters, 1)

	# convert probas to 0/1 predictions
	for i in range(0, a3.shape[1]):
	    if a3[0,i] > 0.5:
	        p[0,i] = 1
	    else:
	        p[0,i] = 0

	# print results

	#print ("predictions: " + str(p[0,:]))
	#print ("true labels: " + str(y[0,:]))
	print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))

	return p
