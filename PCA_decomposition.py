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

def PCA_decomposition(X, number_of_PCs):
    # Standardize attributes
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    pca = PCA(n_components = number_of_PCs).fit(X_standardized)
    X_train_pca = pca.transform(X_standardized)
    print("With {} PCA components {var:0.4f}% of the variance is explained".format(number_of_PCs, var = 100*np.sum(pca.explained_variance_ratio_)))
    
    return X_train_pca

