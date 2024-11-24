import numpy as np
from sklearn.linear_model import LogisticRegression

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc.
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( X_train, y_train ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# X_train has 32 columns containing the challenge bits
	# y_train contains the responses

	feat = my_map(X_train)
	
	classifier = LogisticRegression(C=100,tol=0.01)
	classifier.fit(feat, y_train)
	
	w = classifier.coef_[0]
	b = classifier.intercept_
	
	# THE RETURNED MODEL SHOULD BE A SINGLE VECTOR AND A BIAS TERM
	# If you do not wish to use a bias term, set it to 0
	return w.T, b


################################
# Non Editable Region Starting #
################################
def my_map( X ):
################################
#  Non Editable Region Ending  #
################################

	# Use this method to create features.
	# It is likely that my_fit will internally call my_map to create features for train points

	
    d = np.ones(X.shape) - 2 * X
    feat = []

    for i in range(X.shape[1] - 2, -1, -1):
        d[:, i] = d[:, i+1] * d[:, i]

    for i in range(d.shape[1]):
        for j in range(i + 1, d.shape[1]):
            a = d[:, i] * d[:, j]
            feat.append(a)

    feat.append(d)
    feat = np.column_stack(feat)

    return feat
