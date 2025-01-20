import numpy as np
import pandas as pd
import json
from ase import Atoms

data_dir = r"C:\Users\s201204.FRIDA\Documents\Materials Design with Machine Learning\KaggleCompetition/" # Specify your data path (Folder in which the files are placed)
# Loading the data as pandas DataFrame
test = pd.read_json(data_dir + "test.json")
train = pd.read_json(data_dir + 'train.json')
## Transform atoms entry to ASE atoms object
train.atoms = train.atoms.apply(lambda x: Atoms(**x)) # OBS This one is important!
test.atoms = test.atoms.apply(lambda x: Atoms(**x))

# Look at data
print('Train data shape: {}'.format(train.shape))
train.head()

print('Test data shape: {}'.format(test.shape))
test.head()

# Get more info on data
train.describe()
test.describe()

species = []
number_of_atoms = []
atomic_numbers = []
for atom in pd.concat([train.atoms,test.atoms]):
    species = list(set(species+atom.get_chemical_symbols()))
    atomic_numbers = list(set(atomic_numbers+list(atom.get_atomic_numbers())))
    number_of_atoms.append(len(atom))

max_number_of_atoms = np.max(number_of_atoms)
min_atomic_number = np.min(atomic_numbers)
max_atomic_number = np.max(atomic_numbers)

print(max_number_of_atoms)