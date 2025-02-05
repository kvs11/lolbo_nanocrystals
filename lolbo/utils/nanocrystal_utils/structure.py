from pymatgen.core.structure import Structure, Lattice 
import numpy as np 
import joblib

from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.models.data_utils import *

def get_astr_from_x_tensor(point_cloud, nc_vae_params):
    
    '''
    This function gets chemical information for designed FTCP representations, 
    i.e., formulas, lattice parameters, site fractional coordinates.
    (decoded sampled latent points/vectors).

    Parameters
    ----------
    ftcp_designs : numpy ndarray
        Designed FTCP representations for decoded sampled latent points/vectors.
        The dimensions of the ndarray are number of designs x latent dimension.
    max_elms : int, optional
        Maximum number of components/elements for designed crystals. 
        The default is 3.
    max_sites : int, optional
        Maximum number of sites for designed crystals.
        The default is 20.
    
    '''
    max_elms = nc_vae_params.max_elms
    max_sites = nc_vae_params.max_sites
    zero_pad_rows = nc_vae_params.zero_pad_rows

    # Check if there are zero padding for the x_tensor (PC_array)
    if zero_pad_rows > 0:
        if zero_pad_rows%2 == 0:
            top_pad = bot_pad = zero_pad_rows / 2
        if zero_pad_rows%2 == 1:
            top_pad = int(zero_pad_rows/2)
            bot_pad = top_pad + 1
        
        point_cloud = point_cloud[top_pad:-bot_pad]

    # (from FTCP src) Get the elements_string list 
    ftcp_src_path = '/home/vkolluru/GenerativeModeling/FTCPcode/src'
    elm_str = joblib.load(ftcp_src_path + '/data/element.pkl')

    ftcp_designs = np.expand_dims(point_cloud, axis=0)
    Ntotal_elms = len(elm_str)
    # Get predicted elements of designed crystals
    pred_elm = np.argmax(ftcp_designs[:, :Ntotal_elms, :max_elms], axis=1)
    
    def get_formula(ftcp_designs, ):
        
        # Initialize predicted formulas
        pred_for_array = np.zeros((ftcp_designs.shape[0], max_sites))
        pred_formula = []
        # Get predicted site occupancy of designed crystals
        pred_site_occu = ftcp_designs[:, Ntotal_elms+2+max_sites:Ntotal_elms+2+2*max_sites, :max_elms]
        # Zero non-max values per site in the site occupancy matrix
        temp = np.repeat(np.expand_dims(np.max(pred_site_occu, axis=2), axis=2), max_elms, axis=2)
        pred_site_occu[pred_site_occu < temp]=0
        # Put a threshold to zero empty sites (namely, the sites due to zero padding)
        pred_site_occu[pred_site_occu < 0.05] = 0
        # Ceil the max per site to ones to obtain one-hot vectors
        pred_site_occu = np.ceil(pred_site_occu)
        # Get predicted formulas
        for i in range(len(ftcp_designs)):
            pred_for_array[i] = pred_site_occu[i].dot(pred_elm[i])
            
            if np.all(pred_for_array[i] == 0):
                pred_formula.append([elm_str[0]])
            else:
                temp = pred_for_array[i]
                #temp = temp[:np.where(temp>0)[0][-1]+1]
                # VSCK: Previously, last non-zero is considered as limit. But it is 
                # giving zeros in the middle of occupancy matrix. 
                # So, use first zero as the limit instead
                temp = temp[:np.where(temp==0)[0][0]]
                temp = temp.tolist()
                pred_formula.append([elm_str[int(j)] for j in temp])
        return pred_formula
    
    pred_formula = get_formula(ftcp_designs)[0]
    Nsites = len(pred_formula)
    # Get predicted lattice of designed crystals
    pred_abc = ftcp_designs[0, Ntotal_elms, :3]
    pred_ang = ftcp_designs[0, Ntotal_elms+1,:3]
    # Get predicted site coordinates of designed crystals
    pred_site_coor = ftcp_designs[0, Ntotal_elms+2:Ntotal_elms+2+max_sites, :3]
    pred_site_coor = pred_site_coor[:Nsites, :]

    lattice = Lattice.from_parameters(*pred_abc, *pred_ang)
    astr = Structure(lattice, pred_formula, pred_site_coor, coords_are_cartesian=False)

    return astr
