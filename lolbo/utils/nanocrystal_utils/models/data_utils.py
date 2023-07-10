import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch
import math


########################## Utility functions ################################

def minmax(FTCP):
    '''
    This function performs data normalization for FTCP representation along the second dimension

    Parameters
    ----------
    FTCP : numpy ndarray
        FTCP representation as numpy ndarray.

    Returns
    -------
    FTCP_normed : numpy ndarray
        Normalized FTCP representation.
    scaler : sklearn MinMaxScaler object
        MinMaxScaler used for the normalization.

    '''
    
    dim0, dim1, dim2 = FTCP.shape
    scaler = MinMaxScaler()
    FTCP_ = np.transpose(FTCP, (1, 0, 2))
    FTCP_ = FTCP_.reshape(dim1, dim0*dim2)
    FTCP_ = scaler.fit_transform(FTCP_.T)
    FTCP_ = FTCP_.T
    FTCP_ = FTCP_.reshape(dim1, dim0, dim2)
    FTCP_normed = np.transpose(FTCP_, (1, 0, 2))
    
    return FTCP_normed, scaler

def inv_minmax(FTCP_normed, scaler):
    '''
    This function is the inverse of minmax, 
    which denormalize the FTCP representation along the second dimension

    Parameters
    ----------
    FTCP_normed : numpy ndarray
        Normalized FTCP representation.
    scaler : sklearn MinMaxScaler object
        MinMaxScaler used for the normalization.

    Returns
    -------
    FTCP : numpy ndarray
        Denormalized FTCP representation as numpy ndarray.

    '''
    dim0, dim1, dim2 = FTCP_normed.shape

    FTCP_ = np.transpose(FTCP_normed, (1, 0, 2))
    FTCP_ = FTCP_.reshape(dim1, dim0*dim2)
    FTCP_ = scaler.inverse_transform(FTCP_.T)
    FTCP_ = FTCP_.T
    FTCP_ = FTCP_.reshape(dim1, dim0, dim2)
    FTCP = np.transpose(FTCP_, (1, 0, 2))
    
    return FTCP

# Mean absolute percentage error
def MAPE(y_true, y_pred):
    # Add a small value to avoid division of zero
    y_true, y_pred = np.array(y_true+1e-12), np.array(y_pred+1e-12)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Mean absolute error
def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred), axis=0)

# Mean absolute error for reconstructed site coordinate matrix
def MAE_site_coor(SITE_COOR, SITE_COOR_recon, Nsites):
    site = []
    site_recon = []
    # Only consider valid sites, namely to exclude zero padded (null) sites
    for i in range(len(SITE_COOR)):
        site.append(SITE_COOR[i, :Nsites[i], :])
        site_recon.append(SITE_COOR_recon[i, :Nsites[i], :])
    site = np.vstack(site)
    site_recon = np.vstack(site_recon)
    return np.mean(np.ravel(np.abs(site - site_recon)))

########################## Data Loading Helper Functions ##########################

def load_nanocrystal_train_data(
                    path_to_vae_statedict,
                    num_initialization_points=10_000,
): 
    path_prefix = '/home/vkolluru/GenerativeModeling/Datasets_164x3'
    inp_arr_path = path_prefix + '/L1_Xs.npy'
    y_vals_path = path_prefix + '/L1_Ys.npy'
    embds_path = path_prefix + '/L1_grph_embds.npy'

    input_array = np.load(inp_arr_path, allow_pickle=True).astype('float32')
    y_values_array = np.load(y_vals_path, allow_pickle=True).astype('float32')
    graph_embeds_array = np.load(embds_path, allow_pickle=True).astype('float32')

    # Always compute train_zs from the VAE instead of loading pre-computed 
    # because its consistent when changing different VAE models
    # zs will be computed in NanoCrystalOptimization initialize_objective. 
    zs_from_inputs = None 

    return input_array, graph_embeds_array, zs_from_inputs, y_values_array


def compute_train_zs(
    nanocrystal_objective,
    init_train_x,
    init_train_graph_embds,
    bsz=64
):
    # make sure vae is in eval mode 
    nanocrystal_objective.vae.eval()

    # Do the inference in batches even though the total size is smaller. 
    # Because this way is easily scalable. 
    n_batches = math.ceil(len(init_train_x)/bsz)
    
    init_zs = []
    for i in range(n_batches):
        xs_batch = init_train_x[i*bsz:(i+1)*bsz]
        graph_embds_batch = init_train_graph_embds[i*bsz:(i+1)*bsz]
        zs, _ = nanocrystal_objective.vae_forward(xs_batch, graph_embds_batch)
        init_zs.append(zs.detach().cpu())
    init_zs = torch.cat(init_zs, dim=0)

    return init_zs
