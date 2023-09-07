import fire
from lolbo_nanocrystal.scripts.optimize import Optimize
from lolbo_nanocrystal.scripts.nanocrystal_optimization import NanoCrystalOptimization
from lolbo_nanocrystal.lolbo.nanocrystal_objective import NanoCrystalObjective, NC_VAE_params
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.models.data_utils import load_nanocrystal_train_data, compute_train_zs

import numpy as np

vae_param_dict = {'input_dim': 168, 'graph_embeds_dim': 16, 
                  'max_elms': 1, 'max_sites': 30, 'zero_pad_rows': 3} # check defaults for the rest
nc_vae_params = NC_VAE_params(vae_param_dict)
path_to_vae_ckpt = "/sandbox/vkolluru/Gen_models_for_FANTASTX/test_cases/Si_sw/FX_5000/NC_VAE/lightning_logs/NanoCrystalVAE/version_0/checkpoints/last.ckpt"

#path_prefix = '/home/vkolluru/GenerativeModeling/Datasets_164x3'
#inp_arr_path = path_prefix + '/L1_Xs.npy'
#y_vals_path = path_prefix + '/L1_Ys.npy'
#embds_path = path_prefix + '/L1_grph_embds.npy'

path_prefix = "/sandbox/vkolluru/Gen_models_for_FANTASTX/test_cases/Si_sw/FX_5000/dataset_from_calcs"
init_train_X_path = path_prefix + "/train_arrays/PC_array.npy"
init_train_Y_path= path_prefix + "/train_arrays/Y_array.npy"
init_graph_embeds_path = path_prefix + "/train_arrays/matgl_megnet_embeds.npy"

nc_opt = NanoCrystalOptimization(path_to_vae_ckpt=path_to_vae_ckpt, 
                                 nc_vae_params=nc_vae_params,
                                 init_train_X_path=init_train_X_path,
                                 init_train_Y_path=init_train_Y_path,
                                 init_graph_embeds_path=init_graph_embeds_path,
                                 path_to_energy_yaml="/sandbox/vkolluru/Gen_models_for_FANTASTX/test_cases/Si_sw/FX_5000/Lolbo/energy_FX_Si.yaml",
                                 task_id=1, num_initialization_points=3751, minimize=True)

nc_opt.lolbo_state.update_surrogate_model()
nc_opt.lolbo_state.acquisition()
nc_opt.lolbo_state.update_models_e2e()

"""
# importlib reload
from importlib import reload
import lolbo_nanocrystal.scripts.nanocrystal_optimization
import lolbo_nanocrystal.lolbo.latent_space_objective
import lolbo_nanocrystal.lolbo.nanocrystal_objective
import lolbo_nanocrystal.lolbo.lolbo
nanocrystal_optimization = reload(lolbo_nanocrystal.scripts.nanocrystal_optimization)
latent_space_objective = reload(lolbo_nanocrystal.lolbo.latent_space_objective)
nanocrystal_objective = reload(lolbo_nanocrystal.lolbo.nanocrystal_objective)
lolbo = reload(lolbo_nanocrystal.lolbo.lolbo)
NanoCrystalOptimization = nanocrystal_optimization.NanoCrystalOptimization
NanoCrystalObjective = nanocrystal_objective.NanoCrystalObjective
LatentSpaceObjective = latent_space_objective.LatentSpaceObjective
NC_VAE_params = nanocrystal_objective.NC_VAE_params
"""

#### acquisition()
import torch
from lolbo_nanocrystal.lolbo.utils.bo_utils.turbo import TurboState, update_state, generate_batch
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.models.data_utils import *
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.structure import get_astr_from_x_tensor
import joblib

# 1. Generate a batch of candidates in 
#   trust region using surrogate model
z_next = generate_batch(
    state=nc_opt.lolbo_state.tr_state,
    model=nc_opt.lolbo_state.model,
    X=nc_opt.lolbo_state.train_z,
    Y=nc_opt.lolbo_state.train_y,
    batch_size=nc_opt.lolbo_state.bsz, 
    acqf=nc_opt.lolbo_state.acq_func,
)

# 2. Evaluate the batch of candidates by calling oracle
with torch.no_grad():
    out_dict = nc_opt.lolbo_state.objective(z_next, last_key_idx=len(nc_opt.lolbo_state.train_x_keys))
    z_next = out_dict['valid_zs']
    y_next = out_dict['scores']
    x_next_tensor = out_dict['decoded_xs_tensor']
    x_next_keys = out_dict['x_next_keys']
    graph_embeds_next = out_dict['decoded_xs_graph_embeds']
    if nc_opt.lolbo_state.minimize:
        y_next = y_next * -1

# 3. Add new evaluated points to dataset (update_next)
if len(y_next) != 0:
    y_next = torch.from_numpy(y_next).float()
    x_next_tensor = torch.from_numpy(x_next_tensor).float()
    graph_embeds_next = torch.from_numpy(graph_embeds_next).float()
    nc_opt.lolbo_state.update_next(
        z_next,
        y_next,
        x_next_tensor,
        x_next_keys,
        graph_embeds_next,
        acquisition=True
    )
else:
    nc_opt.lolbo_state.progress_fails_since_last_e2e += 1
    if nc_opt.lolbo_state.verbose:
        print("GOT NO VALID Y_NEXT TO UPDATE DATA, RERUNNING ACQUISITOIN...")