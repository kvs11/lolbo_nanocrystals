"""
Load energy_fx.py module and run this script for debugging any features 
related to lammps energy_code as oracle
"""

import os

from pymatgen.core.structure import Structure


# Make energy code object
energy_input_yaml = 'energy_input.yaml'
energy_code = make_energy_code_object(energy_input_yaml, os.getcwd())


# create an astr of CdTe
cdte_astr = Structure.from_file('POSCAR_CdTe_test')
astr_label = 11


# Run the get_score function. 
energy_code.get_score(cdte_astr, astr_label)