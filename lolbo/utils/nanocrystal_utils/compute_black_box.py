"""
Includes functions related to calculation of DFT/LAMMPS energies of new candidates. 

Methods are inspired from energy.py in FANTASTX
"""
import numpy as np

def get_y_val_from_astr(astr_x, energy_code):
    """
    Args:
    astr_x: Pymatgen structure object corresponding to an input x tensor

    energy_code: lammps_code or vasp_code object (from fantastx)
    
    Returns:
    (float) the objective function value for the input structure 

    """
    y_val = np.random.uniform(0.5, 2.5)
    return y_val

def get_y_vals_from_a_predictor():
    pass