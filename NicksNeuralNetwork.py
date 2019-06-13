########################################################################################################################
"""
GROUP: Required Pacakges
	-Importing packages required for execution

PACKAGES:
	numpy: allows for matrix alegbra which enables matrix representation of neural network for performance increases.
	sys: standard system control package
	sklearn.datasets:provides datasets for testing
	matplotlib: plotting tool
	matplotlib.pyplot: speficic plotting tool
	tkinter: gui control used by matplotlib
	math: for basic math equations used
	warnings: used to implement warnings produced from numpy as errors
	scipy.io: datasets
"""
########################################################################################################################
import sys

import numpy as np

import sklearn.datasets

import matplotlib
#fixes display problem with running through Ubuntu for Windows, no displays will show
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import Tkinter as tk

import math

import warnings
#converts warnings to errors and exits program if not caught
warnings.filterwarnings("error")

import scipy.io

from Regularization import (cost_l2_regularization, backward_propagation_l2_regularization, forward_propagation_droput,
							backward_propagation_dropout)

from Cost import compute_cost

from Propagation import forward_propagation, backward_propagation

from Normalization import z_score_normalization

from Batch import random_mini_batches

from Optimization import initialize_optimizer, update_parameters

from Initialization import initialize_parameters

from Misc import *

from Test import *

########################################################################################################################
"""
GROUP:Neural Network Model
	-Neural Network Model layout

EXTERNAL FUNCTIONS:
					1) model: Defines the neural network model on all the parameters avalible

INTERNAL FUNCTIONS:
					NONE
"""
########################################################################################################################

###EXTERENAL FUNCTIONS###

def model(X, Y, layers_dims, init_type, optimizer_type, learning_rate = 0.0007, lambd = 0, keep_prob = 1,
			mini_batch_size = 64, beta1 = 0.9,beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000, print_cost = True):
	"""
	3-layer neural network model which can be run in different optimizer modes.

	Arguments:
	X -- input data, of shape (number of dimensions, number of examples)
	Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
	layers_dims -- python list, containing the size of each layer
	optimizer_type -- type of optimaztion algorithm the neural network uses for updating weights after backprop
	learning_rate -- the learning rate, scalar.
	lambd -- regularization hyperparameter for l2 regularization, scalar
	keep_prob - probability of keeping a neuron active during drop-out, scalar.
	mini_batch_size -- the size of a mini batch
	beta1 -- Exponential decay hyperparameter for the past gradients estimates
	beta2 -- Exponential decay hyperparameter for the past squared gradients estimates
	epsilon -- hyperparameter preventing division by zero in Adam updates
	num_epochs -- number of epochsbn
	print_cost -- True to print the cost every 1000 epochs

	Returns:
	parameters -- python dictionary containing your updated parameters
	"""

	#Print input shape
	print("***Input Shape***")
	print(X.shape)
	print("")

	#print hyperparameters
	print("***Hyperparameters***")
	print("layers_dims:")
	for index, layer in enumerate(layers_dims):
		print("layer " + str(index) + ":" + str(layer))
	print("init_type:" + init_type)
	print("optimizer_type:" + optimizer_type)
	print("learning_rate:" + str(learning_rate))
	print("lambd:" + str(lambd))
	print("keep_prob:" + str(keep_prob))
	print("mini_batch_size:" + str(mini_batch_size))
	print("beta1:" + str(beta1))
	print("beta2:" + str(beta2))
	print("epsilon:" + str(epsilon))
	print("num_epochs:" + str(num_epochs))
	print("")

	#print start
	print("***Begin Neural Network Training***")

	L = len(layers_dims)             # number of layers in the neural networks
	costs = []                       # to keep track of the cost
	seed=10


	# Initialize parameters
	parameters = initialize_parameters(layers_dims,init_type)

	# Initialize the optimizer
	optimizer = initialize_optimizer(parameters,optimizer_type)

	# Optimization loop
	for i in range(num_epochs):

		# Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
		seed = seed + 1
		minibatches = random_mini_batches(X, Y, mini_batch_size,seed)

		for minibatch in minibatches:

			# Select a minibatch
			(minibatch_X, minibatch_Y) = minibatch

			# Forward propagation
			AL, caches = forward_propagation(minibatch_X, parameters, keep_prob)

			# Compute cost
			cost = compute_cost(AL, minibatch_Y, parameters, lambd)

			# Backward propagation
			grads = backward_propagation(AL, minibatch_Y, caches, lambd, keep_prob)

			# Update parameters
			parameters, optimizer = update_parameters(parameters,
														grads,
														learning_rate,
														optimizer,
														beta1,
														beta2,
														epsilon)

		# Print the cost every 1000 epoch
		if print_cost and i % 1000 == 0:
			print ("Cost after epoch %i: %f" %(i, cost))
		if print_cost and i % 100 == 0:
			costs.append(cost)

	# plot the cost
	plt.plot(costs)
	plt.ylabel('cost')
	plt.xlabel('epochs (per 100)')
	plt.title("Learning rate = " + str(learning_rate))
	plt.show()

	return parameters

########################################################################################################################
"""
GROUP: Python Main

EXTERNAL FUNCTIONS:
					1) main

INTERNAL FUNCTIONS:
					NONE
"""
########################################################################################################################
def main():
	print("NicksNeuralNetwork")
	test_sweep()

if __name__ == '__main__':
	main()
