import numpy as np
import os
import pandas as pd
import json
from dscribe.descriptors import CoulombMatrix
from dscribe.descriptors import MBTR
from ase import Atoms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import root_mean_squared_error as rmse


class GP:
    def __init__(self, train, target, sigma):
        self.train = train
        self.target = target
        self.train_dist = cdist(train, train, metric='sqeuclidean')
        self.target_mean = np.mean(target)
        self.target_normalized = target - self.target_mean
        self.N = len(train)
        self.sigma = sigma

    def kernel(self, x1, x2, ll, k0):
        diff = x1 - x2
        return k0 * np.exp(-np.sum(diff ** 2, axis=-1) / (2 * ll ** 2))
    
    def train_GP(self, ll, k0):
        self.ll = ll
        self.k0 = k0
        K = k0 * np.exp(- self.train_dist / (2 * ll ** 2))
        C = K + k0 * self.sigma ** 2 * np.identity(self.N)
        L = np.linalg.cholesky(C)  # Cholesky decomposition for stability
        self.Cinvt = np.linalg.solve(L.T, np.linalg.solve(L, self.target_normalized))

    def predict(self, fingerprint):
        diff = fingerprint - self.train
        kvec = self.k0 * np.exp(-np.sum(diff ** 2, axis=1) / (2 * self.ll ** 2))
        prediction = np.dot(kvec, self.Cinvt) + self.target_mean
        return prediction