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

# Coulomb Matrices
# Setting up the CM descriptor
def Coulombmatrices(train, max_number_of_atoms):
    cm = CoulombMatrix(n_atoms_max=max_number_of_atoms)

    cmats = np.zeros((len(train),max_number_of_atoms**2))
    for i,atoms in enumerate(train.atoms):
        if i%1000 == 0:
            print(i)
        cmats[i,:] = cm.create(atoms)
    
    return cmats
