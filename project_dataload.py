import numpy as np
import pandas as pd
import json
from ase import Atoms

def data_load(data_dir):
    """Data loading function
    Input:
        data_dir (str): direcory to data folder, relative to current working directory
    Returns: 
        train, test (pd.dataframe): dataframe with info about 2D materials
    """

    # Loading the data as pandas DataFrame
    train = pd.DataFrame(json.load(open(data_dir + "train.json", "rb")))
    test = pd.DataFrame(json.load(open(data_dir + "test.json", "rb")))

    ## Transform atoms entry to ASE atoms object
    train.atoms = train.atoms.apply(lambda x: Atoms(**x)) # OBS This one is important!
    test.atoms = test.atoms.apply(lambda x: Atoms(**x))
    return train, test

def check_data(data):
    """print dimensions of leaded data"""  
    print('Train data shape: {}'.format(data.shape))
    print(data.head())

def summarize_1(data):
    """Get info about species, atomioc numbers, number of atoms in each species. 
    And of inputted species: max number of atoms, min, and max atomic number 
    Input:
        data (pd dataframe): dataframe with 'atoms' column, containing instances of ASE's atoms class
    Returns:
        dict with 3 lists and 3 vals
        species (list of str): atomic species in all materials
        atomic_number (list of int): all atomic numbers represented in materiaæ
        number_of_atoms (list of int): number of atoms for each species
        max_number_of_atoms (int): ...
        min_atomic_number (int): ...
        max_atomic_number (int): ...
    """
    species = []
    number_of_atoms = []
    atomic_numbers = []
    for atom in data.atoms:
        species = list(set(species+atom.get_chemical_symbols()))
        atomic_numbers = list(set(atomic_numbers+list(atom.get_atomic_numbers())))
        number_of_atoms.append(len(atom))

    max_number_of_atoms = np.max(number_of_atoms)
    min_atomic_number = np.min(atomic_numbers)
    max_atomic_number = np.max(atomic_numbers)
    
    dic = {'species': species,
           'atomic_numbers': atomic_numbers,
           'number_of_atoms': number_of_atoms,
           'max_number_of_atoms': max_number_of_atoms,
           'min_atomic_number': min_atomic_number,
           'max_atomic_number': max_atomic_number}
    
    return dic

    


    