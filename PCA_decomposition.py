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

    # Standardize the training set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply PCA to the training set
    pca = PCA(n_components=number_of_PCs) 
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Now apply the same transformations to the test set
    X_test_scaled = scaler.transform(X_test)  # Use the same scaler
    X_test_pca = pca.transform(X_test_scaled)  # Use the same PCA

    return X_train_pca, X_test_pca

