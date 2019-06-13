import numpy as np

########################################################################################################################
"""
GROUP: Normalization
	-Performs Normalization of data layers

	#TODO make this more generalized to provide different normalization techniques on layers
	
EXTERNAL FUNCTIONS: 
					1)z_score_normalization- p
INTERNAL FUNCTIONS:
					
"""
########################################################################################################################
def z_score_normalization(X):
	"""
	preforms z-score normalization on an input layer X

	Arguments:
	X -- layer on which to preform normalization

	Returns:
	X_normalzied - layer normalized
	"""

	#print(X.mean(), X.std())
	#print(np.mean(X,axis=1), np.std(X,axis=1))

	#mean = np.mean(X,axis=1)
	#std = np.std(X,axis=1)

	mean = np.resize(np.mean(X,axis=1),(X.shape[0],X.shape[1]))
	std = np.resize(np.std(X,axis=1),(X.shape[0],X.shape[1]))

	X_normalzied = (X - mean) / (std + 1e-8)

	return X_normalzied