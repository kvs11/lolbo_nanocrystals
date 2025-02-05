
"""
○ Divide each dimension into N=10 equal size chunks
○ Get the maximum and minimum density chunk in each dimension
○ Sample M=100 points per each of max and min density chunk in each dimension
○ Concatenate them
○ Get 100 samples each from max and min density chunks 
○ Decode them and visualize
○ Calculate their energies using LAMMPS
"""
import os, shutil
import numpy as np
import torch

from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.models.IrOx_VAE import NanoCrystalVAE
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.structure import get_astr_from_x_tensor
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.models.data_utils import minmax, inv_minmax
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.energy_fx import make_energy_code_object
from lolbo_nanocrystal.lolbo.nanocrystal_objective import NC_VAE_params


def low_and_high_density_bins(data_samples, n_bins_per_dim):
    # Calculate the minimum and maximum values in each dimension
    min_values = np.min(data_samples, axis=0)
    max_values = np.max(data_samples, axis=0)
    
    # Calculate the dimension ranges
    dimension_ranges = max_values - min_values
    
    # Calculate the bin widths for each dimension
    bin_widths = dimension_ranges / n_bins_per_dim
    
    # Initialize histograms for each dimension
    histograms = []
    for i in range(data_samples.shape[1]):
        bins = np.linspace(min_values[i], max_values[i], n_bins_per_dim + 1)
        histogram, _ = np.histogram(data_samples[:, i], bins=bins)
        histograms.append(histogram)
    
    # Convert histograms to NumPy array for easier manipulation
    histograms = np.array(histograms)
    
    # Find the indices of the maximum and minimum counts in each dimension
    max_counts_indices = np.argmax(histograms, axis=1)
    min_counts_indices = np.argmin(histograms, axis=1)
    
    # Get the bin values corresponding to the maximum and minimum counts
    max_counts_bins = min_values + max_counts_indices * bin_widths
    min_counts_bins = min_values + min_counts_indices * bin_widths
    
    return min_counts_bins, max_counts_bins, bin_widths


def generate_random_points(bin_starts, bin_widths, num_points=10):
    bin_ends = bin_starts + bin_widths

    random_points = []    
    for _ in range(num_points):
        # Generate random values within the hypercube for each dimension
        random_point = [np.random.uniform(bin_starts[dim], bin_ends[dim]) \
                        for dim in range(len(bin_starts))]
        random_points.append(random_point)
    
    return np.array(random_points)


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

num_samples = 20

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


if os.path.exists('high_den_samples') or os.path.exists('low_den_samples'):
    print ('Error: random samples directories already exists!!')
else:
    os.mkdir('high_den_samples')
    os.mkdir('low_den_samples')


low_density_bins, high_density_bins, bin_widths = low_and_high_density_bins(latent_Zs, n_bins_per_dim)

# Sample from high density region
rand_samples_high_den = generate_random_points(high_density_bins, bin_widths, num_points=num_samples)
output_high_den = decode_and_evaluate(rand_samples_high_den, nc_vae, nc_vae_params, scaler_X, energy_code)

# Move calcs dir to high_den_samples
shutil.move('calcs', 'high_den_calcs')
os.mkdir('calcs')

# Sample from low density region 
rand_samples_low_den = generate_random_points(low_density_bins, bin_widths, num_points=num_samples)
output_low_den = decode_and_evaluate(rand_samples_low_den, nc_vae, nc_vae_params, scaler_X, energy_code)

# Move calcs dir to low_den_samples
shutil.move('calcs', 'low_den_calcs')


if save_poscars:

    labels = [i+1 for i in range(num_samples)]
    for label in labels:
        init_astr_hd = output_high_den[label]['init_astr']
        rlxd_astr_hd = output_high_den[label]['relaxed_astr']
        init_astr_ld = output_low_den[label]['init_astr']
        rlxd_astr_ld = output_low_den[label]['relaxed_astr']

        init_astr_hd.to(filename=f'high_den_samples/POSCAR_init_{label}')
        rlxd_astr_hd.to(filename=f'high_den_samples/POSCAR_rlxd_{label}')
        init_astr_ld.to(filename=f'low_den_samples/POSCAR_init_{label}')
        rlxd_astr_ld.to(filename=f'low_den_samples/POSCAR_rlxd_{label}')

if save_as_dict:
    out_scores_hd = {key: {k: v for k, v in subdict.items() if 'score' in k} for key, subdict in output_high_den.items()}
    np.save('high_den_samples/output_scores_hd.npy', out_scores_hd, allow_pickle=True)

    out_scores_ld = {key: {k: v for k, v in subdict.items() if 'score' in k} for key, subdict in output_low_den.items()}
    np.save('low_den_samples/output_scores_ld.npy', out_scores_ld, allow_pickle=True)