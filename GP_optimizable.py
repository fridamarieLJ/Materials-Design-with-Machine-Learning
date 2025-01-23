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

class GP_opt:
    def __init__(self, train, target):
        self.train = train
        self.target = target
        self.train_dist = cdist(train, train, metric='sqeuclidean')
        self.target_mean = np.mean(target)
        self.target_normalized = target - self.target_mean
        self.N = len(train)

    def train_GP(self, ll, k0, sigma):
        self.ll = ll
        self.k0 = k0
        self.sigma = sigma
        K = k0 * np.exp(- self.train_dist / (2 * ll ** 2)) # applying kernel
        C = K + k0 * self.sigma ** 2 * np.identity(self.N)
        L = np.linalg.cholesky(C)  # Cholesky decomposition for stability
        Cinvt = np.linalg.solve(L.T, np.linalg.solve(L, self.target_normalized)) #used in predict
        self.Cinvt = Cinvt
        self.L = L

    def predict(self, fingerprint):
        diff = fingerprint - self.train
        kvec = self.k0 * np.exp(-np.sum(diff ** 2, axis=1) / (2 * self.ll ** 2)) # kernel
        prediction = np.dot(kvec, self.Cinvt) + self.target_mean
        return prediction
    
    def calc_minusloglikelyhood(self):
        """Calculate the minus log likelyhood. 
        For the log of the determinant of a square matrix, utilising the cholsky decomposition, L"""
        singular_values = np.linalg.svd(self.L, compute_uv=False)
        eigenvalues = singular_values**2
        log_eig = np.log(eigenvalues)
        sum_log_eig = np.sum(log_eig)

        term1 = 0.5 * sum_log_eig
        term2 = 0.5 * self.target_normalized.transpose() @ self.Cinvt
        term3 = 0.5 * self.N * np.log(2*np.pi)

        return term1 + term2 + term3