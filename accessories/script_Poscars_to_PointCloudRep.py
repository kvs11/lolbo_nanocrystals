"""
This script takes the path to Poscars, and corresponding data as dict or json file 
And creates PointCloud representations."""

import numpy as np
import joblib, json
from tqdm import tqdm

from pymatgen.core.structure import Structure
from sklearn.preprocessing import OneHotEncoder

# TODO: Use custom method to get OneHot ELM matrix

def get_PC_and_Y_arrays(
        dataset_poscars_path="",
        data_file_path="",
        y_keyword="",
        max_elms="",
        max_sites="",
        zero_pad_rows=0,
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

        # TODO: Automatically determine zero_pad_rows ()= PC.shape[0] %4)
        if zero_pad_rows > 0:
            if zero_pad_rows%2 == 0:
                top_pad = bot_pad = zero_pad_rows / 2
            if zero_pad_rows%2 == 1:
                top_pad = int(zero_pad_rows/2)
                bot_pad = top_pad + 1
            top_zeros = np.zeros((top_pad, max(max_elms, 3)))
            bot_zeros = np.zeros((bot_pad, max(max_elms, 3)))
            PC = np.concatenate((top_zeros, PC, bot_zeros), axis=0)
                
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

if __name__ == "__main__":
    # Variable input parameters needed for this script
    dataset_path = "test_set_poscars"           # path to the dataset_poscars
    data_file_path = "test_set_as_json.json"    # data values file (as_dict.npy or .json) path 
    y_keyword = "total_energy"                  # 'total_energy' or 'formation_energy'
    max_elms = 2                                # Cd and Te
    max_sites = 20                              # Max. no. atoms in any structure
    return_Nsites = False                       
    src_path = '/home/vkolluru/GenerativeModeling/FTCPcode/src' 
    save_all = False 
    zero_pad_rows = 3                           # For the NC-VAE to work seemlessly, we 
                                                # need to make sure out input representation 
                                                # remains such that it remains consistent with 
                                                # convolutions and deconvolutions

    get_PC_and_Y_arrays(
        dataset_poscars_path=dataset_path,
        data_file_path=data_file_path,
        y_keyword=y_keyword,
        max_elms=max_elms,
        max_sites=max_sites,
        zero_pad_rows=zero_pad_rows,
        return_Nsites=return_Nsites,
        save_all=save_all,
        src_path=src_path,
    )