"""
This script takes the path to Poscars, and corresponding data as dict or json file 
And creates PointCloud representations."""

import numpy as np
import joblib, json
from tqdm import tqdm

from pymatgen.core.structure import Structure
from sklearn.preprocessing import OneHotEncoder


# Variable input parameters needed for this script
dataset_path = ""       # path to the dataset_poscars
data_file_path = ""      # data values file (as_dict.npy or .json) path 
max_elms = 3
max_sites = 20 
return_Nsites = False
src_path = None
save_all = False 

# TODO: Use custom method to get OneHot ELM matrix

def get_PC_and_Y_arrays(
        dataset_poscars_path="",
        data_file_path="",
        y_keyword="",
        max_elms="",
        max_sites="",
        return_Nsites=True,
        save_all=False,
        src_path=None,
):
    """
    Method returns the poscars as PointCloud array and corresponding 
    Y values using the provided keyword. Also, save the PC_array and 
    Y_array to file (optional)
    """
    # Read string of elements considered in the study
    try:
        elm_str = joblib.load(src_path + '/data/element.pkl')
    except:
        print ('Provide the correct path to element.pkl and atom_init.json')
    # Build one-hot vectors for the elements
    elm_onehot = np.arange(1, len(elm_str)+1)[:,np.newaxis]
    elm_onehot = OneHotEncoder().fit_transform(elm_onehot).toarray()

    PC_array = []
    Y_array = []
    Nsites = []

    if data_file_path.endswith(".npy"):
        try:
            data_dict = np.load(data_file_path, allow_pickle=True).item()
        except:
            print ("Error in loading the dict.npy type data_file")
    elif data_file_path.endswith(".json"):
        with open(data_file_path) as f:
            data_dict = json.load(f)
    else:
        print ("Error: data_file_path should dict .npy or .json format only")

    poscars = list(data_dict.keys())
    pos_tqdm = tqdm([i for i in range(len(poscars))])
    for idx in pos_tqdm:
        pos_tqdm.set_description('Creating PointCloud representations..')

        poscar = poscars[idx]
        pos_path = dataset_poscars_path + f'/{poscar}'
        crystal = Structure.from_file(pos_path)
        
        # Obtain element matrix
        elm, elm_idx = np.unique(crystal.atomic_numbers, return_index=True)
        # Sort elm to the order of sites in the Poscar
        site_elm = np.array(crystal.atomic_numbers)
        elm = site_elm[np.sort(elm_idx)]

        # Zero pad element matrix to have at least 3 columns
        ELM = np.zeros((len(elm_onehot), max(max_elms, 3),))

        ELM[:, :len(elm)] = elm_onehot[elm-1,:].T

        # Obtain lattice matrix
        latt = crystal.lattice
        LATT = np.array((latt.abc, latt.angles))
        # Zero pad each set of info matrices to have at least 3 columns 
        # and max_elms rows
        LATT = np.pad(LATT, 
                      ((0, 0), (0, max(max_elms, 3)-LATT.shape[1])), 
                      constant_values=0)
        
        # Obtain site coordinate matrix
        SITE_COOR = np.array([site.frac_coords for site in crystal])
        # Pad site coordinate matrix up to max_sites rows and max_elms columns
        SITE_COOR = np.pad(SITE_COOR, 
                           ((0, max_sites-SITE_COOR.shape[0]), 
                            (0, max(max_elms, 3)-SITE_COOR.shape[1])), 
                           constant_values=0)

        # Obtain site occupancy matrix
        # Get the indices of elm that can be used to reconstruct site_elm
        elm_inverse = np.zeros(len(crystal), dtype=int) 
        for count, e in enumerate(elm):
            elm_inverse[np.argwhere(site_elm == e)] = count

        SITE_OCCU = OneHotEncoder().fit_transform(
                            elm_inverse[:,np.newaxis]).toarray()
        SITE_OCCU = np.pad(SITE_OCCU, 
                           ((0, max_sites-SITE_OCCU.shape[0]),
                            (0, max(max_elms, 3)-SITE_OCCU.shape[1])), 
                           constant_values=0)
   
        # Concatenate all matrix sets to create PointCloud representation
        PC = np.concatenate((ELM, LATT, SITE_COOR, SITE_OCCU), axis=0)

        PC_array.append(PC)

        # Get the y_value for this Poscar from data dict
        y_val = data_dict[poscar][y_keyword]
        Y_array.append(y_val)

        # Save Nsites
        Nsites.append(crystal.num_sites)

    PC_array = np.array(PC_array)
    Y_array = np.array(Y_array)
    Nsites = np.array(Nsites)

    if save_all:
        np.save('PC_array.npy', PC_array)
        np.save('Y_array.npy', Y_array)
        np.save('Nsites.npy', Nsites)

    if return_Nsites:
        return PC_array, Y_array, Nsites
    else:
        return PC_array, Y_array


def read_data_file(data_file_path, key):
    """
    Function to read the data file (either a dict.npy or .json format)
    and returns a list/array of the keyword values from the data file
    """
    pass