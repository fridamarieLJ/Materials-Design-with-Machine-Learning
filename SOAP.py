from dscribe.descriptors import SOAP
from ase import Atoms
import numpy as np

def SOAPfingerprints(train, species, r_cut, n_max, l_max, sigma, periodic, average):
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
        periodic=periodic,
        average = 'outer'
    )

    # Pre-allocate storage for SOAP descriptors
    n_features = soap.get_number_of_features()
    
    if average:
        # Average mode: single descriptor per material
        soap_descriptors = np.zeros((len(train), n_features))
    else:
        # Per-atom mode: descriptors for each atom in the largest material
        max_atoms = max(len(atoms) for atoms in train.atoms)
        soap_descriptors = np.zeros((len(train), max_atoms, n_features))

    # Generate SOAP descriptors
    for i, atoms in enumerate(train.atoms):
        if i % 100 == 0:
            print(f"Processing structure {i + 1}/{len(train)}")

        if average:
            # Average SOAP descriptor over all atoms
            soap_descriptors[i, :] = soap.create(atoms, n_jobs=1, average="inner")
        else:
            # SOAP descriptors for each atom in the structure
            per_atom_soap = soap.create(atoms, n_jobs=1)
            soap_descriptors[i, :len(per_atom_soap), :] = per_atom_soap

    return soap_descriptors
