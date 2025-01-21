from dscribe.descriptors import ACSF
from ase import Atoms
import numpy as np

def acsf_finger(train, species, r_cut):
    # Setting up the ACSF descriptor
    acsf = ACSF(
        species= species,
        r_cut= r_cut,
        g2_params=[[1, 1], [1, 2], [1, 3]],
        g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],
    )

    # Pre-allocate storage for SOAP descriptors
    n_features = acsf.get_number_of_features()
    
    # Average mode: single descriptor per material
    acsf_descriptors = np.zeros((len(train), n_features))

    # Generate SOAP descriptors
    for i, atoms in enumerate(train.atoms):
        if i % 100 == 0:
            print(f"Processing structure {i + 1}/{len(train)}")
         # Average SOAP descriptor over all atoms
            acsf_descriptors[i, :] = acsf.create(atoms, n_jobs=1)

    return acsf_descriptors