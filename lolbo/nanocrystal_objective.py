import os
import numpy as np
import torch 
from operator import itemgetter


from lolbo_nanocrystal.lolbo.latent_space_objective import LatentSpaceObjective
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.models.IrOx_VAE import NanoCrystalVAE
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.compute_black_box import get_y_val_from_astr
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.energy_fx import initialize_energy_code
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.fingerprinting import Comparator

class NanoCrystalObjective(LatentSpaceObjective):
    '''MoleculeObjective class supports all molecule optimization
        tasks and uses the SELFIES VAE by default '''

    def __init__(
        self,
        path_to_vae_statedict: str=None,
        path_to_vae_ckpt: str=None,
        fp_label: str='bag-of-bonds',
        fp_tolerances=None,
        energy_input_yaml=None,
        pool_dict={},
        labels_count=0,
        num_calls=0,
    ):

        self.dim                    = 32 # NanoCrystal VAE DEFAULT LATENT SPACE DIM
        self.path_to_vae_statedict  = path_to_vae_statedict # path to trained vae stat dict
        self.path_to_vae_ckpt = path_to_vae_ckpt
        self.fp_label = fp_label
        self.fp_tolerances = fp_tolerances

        self.energy_code = initialize_energy_code.make_energy_code_object(
                                                    energy_input_yaml, os.getcwd())

        super().__init__(
            num_calls=num_calls,
            pool_dict=pool_dict,
            labels_count=labels_count,
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


    def query_oracle(self, x):
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
        score = get_y_val_from_astr(x) 

        return score


    def initialize_vae(self):
        ''' Sets self.vae to the desired pretrained vae and 
            sets self.dataobj to the corresponding data class 
            used to tokenize inputs, etc. '''
        self.vae = NanoCrystalVAE()
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
        
        z = z.reshape(-1,self.dim)
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
