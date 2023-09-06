"""
Script that uses a pre-trained VAE and gives the latent vectors for the 
input dataset using the encoder.
"""
import torch
import math
import numpy as np

from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.models.IrOx_VAE import NanoCrystalVAE



def get_latent_Zs(nc_vae_params, input_x_path, input_graph_embds_path, bsz, 
                  path_to_vae_ckpt=None, path_to_vae_statedict=None,
                  save_to_npy_file=False):

    nc_vae = NanoCrystalVAE(**nc_vae_params)

    # load in state dict of trained model:
    if path_to_vae_ckpt:
        checkpoint = torch.load(path_to_vae_ckpt) 
        nc_vae.load_state_dict(checkpoint['state_dict'], strict=True)
    elif path_to_vae_statedict:
        state_dict = torch.load(path_to_vae_statedict) 
        nc_vae.load_state_dict(state_dict, strict=True)

    nc_vae.eval()

    # Load inputs and graph embeds
    input_x = np.load(input_x_path, allow_pickle=True).astype('float32')
    input_x = torch.tensor(input_x)

    input_graph_embds = np.array([[]] * input_x.shape[0]).astype('float32')
    if input_graph_embds_path is not None:
        input_graph_embds = np.load(input_graph_embds_path, 
                                allow_pickle=True).astype('float32')
    input_graph_embds = torch.tensor(input_graph_embds)
    
    # Do the inference in batches even though the total size is smaller. 
    # Because this way is easily scalable. 
    n_batches = math.ceil(len(input_x)/bsz)

    latent_zs = []
    for i in range(n_batches):
        xs_batch = input_x[i*bsz:(i+1)*bsz]
        graph_embds_batch = input_graph_embds[i*bsz:(i+1)*bsz]
        out_dict = nc_vae(xs_batch, graph_embds_batch)
        zs_batch = out_dict['z']
        latent_zs.append(zs_batch.detach().cpu())
    latent_zs = torch.cat(latent_zs, dim=0)
    
    if save_to_npy_file:
        np.save("latent_Zs.npy", latent_zs)
    return latent_zs


if __name__ == "__main__":
    path_prefix = '/sandbox/vkolluru/Gen_models_for_FANTASTX/CdTe_test_case/Fantastx_GA_rand30/dataset_from_calcs'
    train_PC_array_path = path_prefix + '/train_set_arrays/PC_array.npy'
    train_embds_path = path_prefix + '/train_set_arrays/matgl_megnet_embeds.npy'
    
    nc_vae_params = dict(
        input_dim = 148,
        channel_dim = 3,
        regression_dim = 1,
        graph_embds_dim = 0,
        latent_dim = 32,
        max_filters = 128,
        filter_size = [5, 3, 3],
        strides = [2, 2, 1],
        coeffs = (1, 2, 15),
    )

    input_x_path = train_PC_array_path
    input_graph_embds_path = None #train_embds_path
    bsz = 64
    path_to_vae_ckpt = "/sandbox/vkolluru/Gen_models_for_FANTASTX/CdTe_test_case/Fantastx_GA_rand30/dataset_from_calcs/NCVAE_model_train/lightning_logs/NanoCrystalVAE/version_3/checkpoints/last.ckpt"
    path_to_vae_statedict = None
    save_to_npy_file = True
    
    get_latent_Zs(nc_vae_params=nc_vae_params, 
                  input_x_path=input_x_path, 
                  input_graph_embds_path=input_graph_embds_path, 
                  bsz=bsz, 
                  path_to_vae_ckpt=path_to_vae_ckpt, 
                  path_to_vae_statedict=path_to_vae_statedict, 
                  save_to_npy_file=save_to_npy_file)