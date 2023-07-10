import numpy as np
import torch 

from lolbo_nanocrystal.lolbo.latent_space_objective import LatentSpaceObjective
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.models.IrOx_VAE import NanoCrystalVAE
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.compute_black_box import get_y_val_from_input

class NanoCrystalObjective(LatentSpaceObjective):
    '''MoleculeObjective class supports all molecule optimization
        tasks and uses the SELFIES VAE by default '''

    def __init__(
        self,
        path_to_vae_statedict: str=None,
        path_to_vae_ckpt: str=None,
        xs_to_scores_dict={},
        num_calls=0,
    ):

        self.dim                    = 32 # NanoCrystal VAE DEFAULT LATENT SPACE DIM
        self.path_to_vae_statedict  = path_to_vae_statedict # path to trained vae stat dict
        self.path_to_vae_ckpt = path_to_vae_ckpt

        super().__init__(
            num_calls=num_calls,
            xs_to_scores_dict=xs_to_scores_dict,
            # task_id=task_id,
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
        # sample molecular string form VAE decoder
        decoded_sample = self.vae.sample(z=z.reshape(-1, 2, 128))

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
        score = get_y_val_from_input(x) 

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
        # assumes xs_batch is a batch of smiles strings 
        dict = self.vae(xs_batch.cuda(), graph_embds_batch.cuda())
        vae_loss, z = dict['loss'], dict['z']
        z = z.reshape(-1,self.dim)

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
