import numpy as np
import os
import pandas as pd
import json
from dscribe.descriptors import CoulombMatrix
from ase import Atoms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt


def PCA_decomposition(X_train, X_test, number_of_PCs):

    # Apply PCA to the training set
    pca = PCA(n_components=number_of_PCs).fit(X_train)
    X_train_pca = pca.transform(X_train)
    print("With {} PCA components {var:0.4f}% of the variance is explained".format(number_of_PCs, var = 100*np.sum(pca.explained_variance_ratio_)))
    print('X_train_pca: {}'.format(X_train_pca.shape))

    # Now apply the same transformations to the test set
    X_test_pca = pca.transform(X_test)

    print('X_test_pca: {}'.format(X_test_pca.shape))

    return X_train_pca, X_test_pca
