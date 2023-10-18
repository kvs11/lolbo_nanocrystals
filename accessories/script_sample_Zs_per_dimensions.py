
"""
This script takes a set of data points, finds the minimum and maximum values for
each dimension, and then generates samples ensuring that one dimension varies 
incrementally within its min and max values while fixing the others.
"""
import os, shutil
import numpy as np
import torch

from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.models.IrOx_VAE import NanoCrystalVAE
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.structure import get_astr_from_x_tensor
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.models.data_utils import minmax, inv_minmax
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.energy_fx import make_energy_code_object
from lolbo_nanocrystal.lolbo.nanocrystal_objective import NC_VAE_params


def generate_samples_sequential(latent_Zs, n_samples):
    """
    finds the minimum and maximum values for each dimension, 
    and then generates samples ensuring that one dimension varies 
    incrementally within its min and max values while fixing the others.
    """
    n_Z = latent_Zs.shape[1]
    min_vals = np.min(latent_Zs, axis=0)
    max_vals = np.max(latent_Zs, axis=0)
    
    samples = np.zeros((n_Z, n_samples, n_Z))
    
    main_sample = np.random.uniform(min_vals, max_vals, n_Z)

    for i in range(n_Z):
        for j in range(n_samples):
            sample = main_sample.copy()
            sample[i] = np.linspace(min_vals[i], max_vals[i], n_samples)[j]
            samples[i, j, :] = sample
    
    return samples


def decode_and_evaluate(random_samples, nc_vae, nc_vae_params, scaler_X, energy_code):
    labels = [i+1 for i in range(len(random_samples))]

    # Get the decoded samples using VAE decoder
    random_samples = torch.tensor(random_samples).float().cuda()
    decoded_samples = nc_vae.decoder(random_samples)

    # descale the decoded tensor to get PC representation
    descaled_decoded_samples = inv_minmax(decoded_samples.detach().cpu(), scaler_X)


    output_dict = {}
    for label, x_descaled in zip(labels, descaled_decoded_samples):
        astr_x = get_astr_from_x_tensor(x_descaled, nc_vae_params)
        init_astr, rlxd_astr, init_score, rlxd_score = energy_code.get_score(astr_x, label)
        output_dict[label] = {'init_astr': init_astr,
                              'relaxed_astr': rlxd_astr,
                              'init_score': init_score,
                              'relaxed_score': rlxd_score}

    return output_dict


## Load data
path_prefix = '/sandbox/vkolluru/Gen_models_for_FANTASTX/test_cases/CdTe_case/FX_5000/Datasets/first_500'
train_PC_array_path = path_prefix + '/train_set_arrays/PC_array.npy'
#train_Y_array_path = path_prefix + '/train_set_arrays/Y_array.npy'
#train_embds_path = path_prefix + '/train_set_arrays/matgl_megnet_embeds.npy'
path_to_energy_yaml = '/sandbox/vkolluru/Gen_models_for_FANTASTX/test_cases/CdTe_case/FX_5000/VAEs_training/first_500/energy_FX_CdTe.yaml'

latent_Zs_path = "latent_Zs.npy"
n_bins_per_dim = 10


path_to_vae_ckpt = "/sandbox/vkolluru/Gen_models_for_FANTASTX/test_cases/CdTe_case/FX_5000/VAEs_training/tune_vae_on_f_500/m3/lightning_logs/NanoCrystalVAE/version_0/checkpoints/last.ckpt"
path_to_vae_statedict = None

save_poscars = True
save_as_dict = True

n_samples_per_dim = 10


# NC-VAE params
vae_params = dict(
    input_dim = 188,
    channel_dim = 3,
    regression_dim = 1,
    graph_embds_dim = 16,
    latent_dim = 32,
    max_filters = 128,
    filter_size = [5, 3, 3],
    strides = [2, 2, 1],
    coeffs = (1, 2, 15),
)
vae_params_ext = vae_params.copy()
vae_params_ext.update(dict(max_elms = 2, max_sites = 40, zero_pad_rows = 3,))

#############


PC_array = np.load(train_PC_array_path, allow_pickle=True).astype('float32')
_, scaler_X = minmax(PC_array)
# Y = np.load(Y_array_path, allow_pickle=True).astype('float32')
latent_Zs = np.load(latent_Zs_path, allow_pickle=True).astype('float32')

energy_code = make_energy_code_object(path_to_energy_yaml, os.getcwd())

# Initiate and load NC VAE
nc_vae = NanoCrystalVAE(**vae_params)

# load in state dict of trained model:
if path_to_vae_ckpt:
    checkpoint = torch.load(path_to_vae_ckpt) 
    nc_vae.load_state_dict(checkpoint['state_dict'], strict=True)
elif path_to_vae_statedict:
    state_dict = torch.load(path_to_vae_statedict) 
    nc_vae.load_state_dict(state_dict, strict=True)

nc_vae.eval()
nc_vae.cuda()

# create NC_VAE_params class instance
nc_vae_params = NC_VAE_params(vae_params_ext)


if os.path.exists('rand_samples_sequential'):
    print ('Error: rand_samples_sequential directory already exists!!')
else:
    os.mkdir('rand_samples_sequential')

os.mkdir('calcs')

# Sample from low density region 
rand_samples = generate_samples_sequential(latent_Zs=latent_Zs, n_samples=n_samples_per_dim)
rand_samples = np.reshape(rand_samples, (-1, 32))
output_dict = decode_and_evaluate(rand_samples, nc_vae, nc_vae_params, scaler_X, energy_code)

# Move calcs dir to low_den_samples
shutil.move('calcs', 'rand_samples_sequential')


if save_poscars:

    labels = output_dict.keys()
    for label in labels:
        init_astr_hd = output_dict[label]['init_astr']
        rlxd_astr_hd = output_dict[label]['relaxed_astr']

        init_astr_hd.to(filename=f'rand_samples_sequential/POSCAR_init_{label}')
        rlxd_astr_hd.to(filename=f'rand_samples_sequential/POSCAR_rlxd_{label}')

if save_as_dict:
    out_scores_hd = {key: {k: v for k, v in subdict.items() if 'score' in k} \
                     for key, subdict in output_dict.items()}
    np.save('rand_samples_sequential/output_scores_hd.npy', out_scores_hd, allow_pickle=True)