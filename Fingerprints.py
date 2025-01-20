import numpy as np
import pandas as pd
from dscribe.descriptors import CoulombMatrix
from ase import Atoms

# Coulomb Matrices
# Setting up the CM descriptor
def Coulombmatrices(train, max_number_of_atoms):
    cm = CoulombMatrix(n_atoms_max=max_number_of_atoms, permutation = 'sorted_l2')

    cmats = np.zeros((len(train),max_number_of_atoms**2))
    for i,atoms in enumerate(train.atoms):
        if i%1000 == 0:
            print(i)
        cmats[i,:] = cm.create(atoms)
    
    return cmats

