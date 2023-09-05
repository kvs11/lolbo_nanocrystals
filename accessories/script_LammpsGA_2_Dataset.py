"""
Script to create a dataset (POSCAR, value) from the LAMMPS 
calculations of candidates within a FANTASTX run.
"""
from sklearn.model_selection import train_test_split
import numpy as np
import json
import os

from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.energy_fx import lammps_code
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.fingerprinting import Comparator

from fx19 import structure_record


# variables to be given as input
element_symbols = ['Cd', 'Te']                    # all elements present in the system
fp_label = 'bag-of-bonds'                   # fingerprinting mode
fp_tolerances = {fp_label: [0.04, 0.7]}     # tolerances for the fingerprinting mode
main_path = '/sandbox/vkolluru/Gen_models_for_FANTASTX/test_cases/CdTe_sw/FX_2000'
calcs_dir_path = main_path + '/calcs'       # path to calcs dir
max_n_for_vae = 500                         # use first 500 candidate lammps relaxations 
dump_file_name = 'rlx.str'                  # dump file name
test_set_size = 0.2                         # float or int
dataset_dir = main_path + '/dataset_from_calcs'           # path to create dataset dir
last_only = False                           # True to collect only final relaxed structures
save_data_as_dict = True 
save_data_as_json = True

# This is the dataset_dir structure:
# dirs: train_set_poscars/ , test_set_poscars/
# files: train_set_as_dict.npy, train_set_as_json.json
# files: test_set_as_dict.npy, test_set_as_json.json 

# what information is contained within the data dict
# Eg entry in data_dict: 
"""
'POSCAR_21' is the key to its data in the dict. Further, 
'POSCAR_21' is the filename of its structure in train_set_poscars dir
{
    'POSCAR_21': {
        'total_energy': -10,
        'comp_dict': {
            'Cd': 5,
            'Te': 4
        },
        'total_atoms': 9,
        'epa': -1.1111,
        <custom_key>: <custom_value>
    }
}
"""
custom_keys = []                # custom_keys list to be added to data dict
custom_funcs = []               # functions list to calculate custom_vals 
                                # corresponding to the custom_keys

# Example custom function to calculate the formation energies of CdTe 
# structures with a user-provided mu_Cd and mu_Te values
def form_en_func(*args):
    '''
    The custom function to calculate formation energy. 
    The inputs should be the list of keys within the data dict
    '''
    tot_en, n_Cd, n_Te = args
    mu_Cd, mu_Te = 1.16, 2.19
    return tot_en - n_Cd*mu_Cd - n_Te*mu_Te


# Create dataset from lammps relaxations

# create dataset_directory 
os.mkdir(dataset_dir)#, exist_ok=True)
os.mkdir(dataset_dir+'/train_set_poscars')#, exist_ok=True)
os.mkdir(dataset_dir+'/test_set_poscars')#, exist_ok=True)

# Go to calcs dir and get all model dirs
if max_n_for_vae is not None:
    with open(main_path + '/data_file') as f:
        lines = f.readlines()
        lines = lines[2:]
    all_model_dirs = [l.split()[0] for l in lines[:max_n_for_vae]]
else:
    all_model_dirs = [i_dir for i_dir in os.listdir(calcs_dir_path) \
                    if os.path.isdir(calcs_dir_path+f'/{i_dir}')]

all_astrs, all_tot_ens = [], []

for i_dir in all_model_dirs:
    rlx_astr_path = f'{calcs_dir_path}/{i_dir}/relax/{dump_file_name}'
    data_in_path = f'{calcs_dir_path}/{i_dir}/relax/in.data'
    log_lammps_path = f'{calcs_dir_path}/{i_dir}/relax/log.lammps'
    
    astrs, step_inds = lammps_code.get_relaxed_cell(
        rlx_astr_path, data_in_path, element_symbols, last_only=last_only)
    step_tot_ens = lammps_code.get_step_energies(log_lammps_path, step_inds)

    all_astrs += astrs
    all_tot_ens += step_tot_ens

# Shuffle all_astrs and all_tot_ens to avoid favoring initial steps within 
# each relaxation during duplicate detection in next step.
temp_all = list(zip(all_astrs, all_tot_ens))
random.shuffle(temp_all)
all_astrs, all_tot_ens = zip(*temp_all)

# Delete duplicate or redundant structures 
reg_id = structure_record.register_id()
comparator = Comparator(label=fp_label, tolerances=fp_tolerances)
uniq_astr_inds, uniq_astrs, uniq_models = [], [], []
for i, astr in enumerate(all_astrs):
    model_i = structure_record.model(astr, reg_id)
    comparator.create_fingerprint(model_i)
    if i == 0:
        uniq_astr_inds.append(i)
        uniq_astrs.append(astr)
        uniq_models.append(model_i)
        continue
    uniq = comparator.check_model_uniqueness(model_i, uniq_models)
    if uniq:
        uniq_astr_inds.append(i)
        uniq_astrs.append(astr)
        uniq_models.append(model_i)

uniq_tot_ens = [all_tot_ens[i] for i in uniq_astr_inds]
print ("Total structures: {}\n\
       fp tolerances: {}\n\
       Unique structures: {}".format(
                        len(all_astrs), fp_tolerances, len(uniq_tot_ens)))

# Do the test train split
train_astrs, test_astrs, train_totens, test_totens = train_test_split(
            uniq_astrs, uniq_tot_ens, test_size=test_set_size, shuffle=True)

# Save Train dataset and Test dataset separately
for set_type in ['train', 'test']:
    if set_type == 'train':
        set_astrs, set_totens = train_astrs, train_totens
    else:
        set_astrs, set_totens = test_astrs, test_totens
    full_data = {}
    for i, (astr, toten) in enumerate(zip(set_astrs, set_totens)):
        pos_key = f'POSCAR_{i}'
        pos_dict = {}
        pos_dict['total_energy'] = toten
        pos_dict['comp_dict'] = astr.composition.as_dict()
        pos_dict['total_atoms'] = int(sum(pos_dict['comp_dict'].values()))
        pos_dict['epa'] = toten / pos_dict['total_atoms']
        #pos_dict['formation_energy'] = form_en_func(toten, 
        #                                            pos_dict['comp_dict']['Cd'],
        #                                            pos_dict['comp_dict']['Te'])
        full_data[pos_key] = pos_dict

        # save poscars
        astr.to(filename=f'{dataset_dir}/{set_type}_set_poscars/{pos_key}')
        # save data as dict or json
        if save_data_as_dict:
            np.save(f'{dataset_dir}/{set_type}_set_as_dict.npy', full_data)
        if save_data_as_json:
            with open(f'{dataset_dir}/{set_type}_set_as_json.json', 'w') as f:
                json.dump(full_data, f, indent=4)

with open(dataset_dir+ '/params_of_dataset.txt', 'w') as f:
    par_lines = [f"fp_label = {fp_label}\n",
                 f"fp_tolerances = {fp_tolerances}\n",
                 f"max_n_for_vae = {max_n_for_vae}\n",
                 f"test_set_size = {test_set_size}\n",
                 f"last_only = {last_only}\n"]
    f.writelines(par_lines)