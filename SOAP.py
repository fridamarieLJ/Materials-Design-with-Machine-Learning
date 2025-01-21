from dscribe.descriptors import SOAP
from ase import Atoms
import numpy as np

def SOAPfingerprints(train, species, r_cut, n_max, l_max, sigma):
    """
    Generate SOAP descriptors for a dataset of atomic structures.

    Parameters:
    - train: Dataset of atomic structures (e.g., ASE Atoms objects).
    - species: List of atomic species present in the dataset.
    - r_cut: Cutoff radius for SOAP.
    - n_max: Number of radial basis functions.
    - l_max: Maximum degree of spherical harmonics.
    - sigma: Width of the Gaussian used for atom density smoothing.
    - periodic: Boolean, whether to consider periodic boundary conditions.
    - average: Boolean, whether to average the SOAP descriptor over all atoms.

    Returns:
    - soap_descriptors: Numpy array of SOAP fingerprints.
    """
    # Initialize SOAP descriptor
    soap = SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        sigma=sigma,
        periodic= True,
        average = "outer"
    )

    # Pre-allocate storage for SOAP descriptors
    n_features = soap.get_number_of_features()
    
    # Average mode: single descriptor per material
    soap_descriptors = np.zeros((len(train), n_features))

    # Generate SOAP descriptors
    for i, atoms in enumerate(train.atoms):
        if i % 100 == 0:
            print(f"Processing structure {i + 1}/{len(train)}")
         # Average SOAP descriptor over all atoms
            soap_descriptors[i, :] = soap.create(atoms, n_jobs=1)

    return soap_descriptors
