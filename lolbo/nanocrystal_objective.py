import os
import numpy as np
import torch 
from operator import itemgetter
from collections import OrderedDict
from typing import List, Callable, Union, Any, TypeVar, Tuple

import matgl

from lolbo_nanocrystal.lolbo.latent_space_objective import LatentSpaceObjective
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.models.IrOx_VAE import NanoCrystalVAE
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.compute_black_box import get_y_val_from_astr
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.energy_fx import make_energy_code_object
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.fingerprinting import Comparator
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.megnet_torch import MEGNetShortModel

class NanoCrystalObjective(LatentSpaceObjective):
    '''MoleculeObjective class supports all molecule optimization
        tasks and uses the SELFIES VAE by default '''

    def __init__(
        self,
        path_to_vae_statedict: str=None,
        path_to_vae_ckpt: str=None,
        nc_vae_params=None,
        scaler_X=None,
        scaler_Y=None,
        fp_label: str='bag-of-bonds',
        fp_tolerances=None,
        path_to_energy_yaml=None,
        pool_dict={},
        labels_count=0,
        num_calls=0,
    ):

        self.vae_latent_dim                    = 32 # NanoCrystal VAE default latent dim
        self.path_to_vae_statedict  = path_to_vae_statedict # path to trained vae stat dict
        self.path_to_vae_ckpt = path_to_vae_ckpt
        self.vae_params = nc_vae_params
        #self.scaler_X = scaler_X
        #self.scaler_Y = scaler_Y
        self.fp_label = fp_label
        self.fp_tolerances = fp_tolerances

        self.energy_code = make_energy_code_object(path_to_energy_yaml, os.getcwd())

        super().__init__(
            num_calls=num_calls,
            pool_dict=pool_dict,
            labels_count=labels_count,
            nc_vae_params=nc_vae_params,
            scaler_X=scaler_X,
            scaler_Y=scaler_Y,
            energy_code=self.energy_code,
        )


    def vae_decode(self, z):
        '''Input
                z: a tensor latent space points
            Output
                a corresponding list of the decoded input space 
                items output by vae decoder 
        '''
        if type(z) is np.ndarray: 
            z = torch.from_numpy(z).float()
        z = z.cuda()
        self.vae = self.vae.eval()
        self.vae = self.vae.cuda()
        # decoded Xs samples form VAE decoder
        decoded_sample = self.vae.decoder(z)

        return decoded_sample


    def query_oracle(self, astr_x, astr_label):
        ''' Input: 
                a single input space item x
            Output:
                method queries the oracle and returns 
                the corresponding score y,
                or np.nan in the case that x is an invalid input
        '''
        # method to 
        # --> convert FTCP to VASP inputs
        # Runs VASP and returns the objective function value (eg. formation energy)
        score = self.energy_code.get_score(astr_x, astr_label)
        #score = get_y_val_from_astr(astr_x, astr_label, self.energy_code)

        return score


    def initialize_vae(self):
        ''' Sets self.vae to the desired pretrained vae and 
            sets self.dataobj to the corresponding data class 
            used to tokenize inputs, etc. '''
        self.vae = NanoCrystalVAE(
                        input_dim=self.vae_params.input_dim,
                        channel_dim=self.vae_params.channel_dim,
                        regression_dim=self.vae_params.regression_dim,
                        graph_embds_dim=self.vae_params.graph_embds_dim,
                        coeffs=self.vae_params.coeffs,
                        latent_dim=self.vae_params.latent_dim,
                        max_filters=self.vae_params.max_filters,
                        filter_size=self.vae_params.filter_size,
                        strides=self.vae_params.strides,
                    )
        # load in state dict of trained model:
        if self.path_to_vae_ckpt:
            checkpoint = torch.load(self.path_to_vae_ckpt) 
            self.vae.load_state_dict(checkpoint['state_dict'], strict=True)
        elif self.path_to_vae_statedict:
            state_dict = torch.load(self.path_to_vae_statedict) 
            self.vae.load_state_dict(state_dict, strict=True)
        self.vae = self.vae.cuda()
        self.vae = self.vae.eval()

    def initialize_comparator(self):
        ''' Sets variable self.comparator to the Comparator class with appropriate parameters 
            The Comparator should be connected with FANTASTX iin future. 
            For dev purposes, fingerprinting.py is copied from fantastx entirely. 
            (Currently using bag-of-bonds with respective thresholds) '''
        self.comparator = Comparator(label=self.fp_label, tolerances=self.fp_tolerances)

    def initialize_megnet_short_model(self, pre_trained_matgl_model):
        """
        Given the name of a pre-trained matgl model, this function creates another 
        model with same weights to provide the second last layer embeddings
        """
        megnet_model = matgl.load_model(pre_trained_matgl_model)
        megnet_short_model = MEGNetShortModel()

        megnet_sd = megnet_model.model.state_dict()
        short_model_keys = megnet_short_model.state_dict().keys()

        megnet_sd_for_short_model = OrderedDict()
        for key in short_model_keys:
            megnet_sd_for_short_model[key] = megnet_sd[key]

        megnet_short_model.load_state_dict(megnet_sd_for_short_model)

        self.megnet_short_model = megnet_short_model

    def vae_forward(self, xs_batch, graph_embds_batch):
        ''' Input: 
                a list xs 
            Output: 
                z: tensor of resultant latent space codes 
                    obtained by passing the xs through the encoder
                vae_loss: the total loss of a full forward pass
                    of the batch of xs through the vae 
                    (ie reconstruction error)
        '''
        def list_or_np_to_tensor(xx):
            np_arr = np.array(xx, dtype='float32')
            xx = torch.from_numpy(np_arr)
            del np_arr
            return xx
        
        # convert xs_batch and graph_embeds to tensors if needed
        if not torch.is_tensor(xs_batch):
            xs_batch = list_or_np_to_tensor(xs_batch)
        if not torch.is_tensor(graph_embds_batch):
            graph_embds_batch = list_or_np_to_tensor(graph_embds_batch)

        outputs_dict = self.vae(xs_batch.cuda(), graph_embds_batch.cuda())

        # Compute loss here (same as in pl training_step)
        # NOTE: Use dummy ys that computes regression branch loss because 
        # we don't use it. Only Recon and KLD losses are used.
        z, mu_, log_var_, reg_output_, reconstructed_output_ = itemgetter(
            'z', 'mu', 'log_var', 'reg_output', 'reconstructed_output')(outputs_dict)
        
        z = z.reshape(-1,self.vae_latent_dim)
        dummy_ys = torch.ones_like(reg_output_)
        
        # Compute the loss and its gradients
        loss_fn_args = [reconstructed_output_.cuda(), xs_batch.cuda(), 
                        mu_.cuda(), log_var_.cuda(), 
                        reg_output_.cuda(), dummy_ys.cuda()]
        loss_fn_kwargs = {
            'coeff_recon': self.vae.hparams.coeffs[0],
            'coeff_KL': self.vae.hparams.coeffs[1], 
            'coeff_reg': self.vae.hparams.coeffs[2]
        }
        loss_dict = NanoCrystalVAE.loss_function(*loss_fn_args, **loss_fn_kwargs)

        recon_loss = loss_dict['Reconstruction_loss'] 
        kld_loss = loss_dict['KLD_loss']
        vae_loss = torch.mean(self.vae.hparams.coeffs[0] * recon_loss + \
                              self.vae.hparams.coeffs[1] * kld_loss)
        
        return z, vae_loss


class NC_VAE_params:

    def __init__(self, vae_params_dict):

        self.input_dim: int = 164
        self.channel_dim: int = 3 
        self.regression_dim: int = 1
        self.graph_embds_dim: int = 16
        self.coeffs: Tuple = (1, 2, 10,)
        self.latent_dim: int = 32
        self.max_filters: int = 128
        self.filter_size: List = [5, 3, 3]
        self.strides: List = [2, 2, 1]
        self.max_elms: int = 2
        self.max_sites: int = 30
        self.zero_pad_rows: int = 0

        if 'input_dim' in vae_params_dict:
            self.input_dim = vae_params_dict['input_dim']
        if 'channel_dim' in vae_params_dict:
            self.channel_dim = vae_params_dict['channel_dim']
        if 'regression_dim' in vae_params_dict:
            self.regression_dim = vae_params_dict['regression_dim']
        if 'graph_embds_dim' in vae_params_dict:
            self.graph_embds_dim = vae_params_dict['graph_embds_dim']
        if 'coeffs' in vae_params_dict:
            self.coeffs = vae_params_dict['coeffs']
        if 'latent_dim' in vae_params_dict:
            self.latent_dim = vae_params_dict['latent_dim']
        if 'max_filters' in vae_params_dict:
            self.max_filters = vae_params_dict['max_filters']
        if 'filter_size' in vae_params_dict:
            self.filter_size = vae_params_dict['filter_size']
        if 'strides' in vae_params_dict:
            self.strides = vae_params_dict['strides']
        if 'max_elms' in vae_params_dict:
            self.max_elms = vae_params_dict['max_elms']
        if 'max_sites' in vae_params_dict:
            self.max_sites = vae_params_dict['max_sites']
        if 'zero_pad_rows' in vae_params_dict:
            self.zero_pad_rows = vae_params_dict['zero_pad_rows']


if __name__ == "__main__":
    # testing molecule objective
    obj1 = NanoCrystalObjective() 
    print ("*** Test Test Test ***")
    print(obj1.num_calls)
    dict1 = obj1(torch.randn(10,256))
    print(dict1['scores'], obj1.num_calls)
    dict1 = obj1(torch.randn(3,256))
    print(dict1['scores'], obj1.num_calls)
    dict1 = obj1(torch.randn(1,256))
    print(dict1['scores'], obj1.num_calls)
