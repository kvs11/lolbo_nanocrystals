import numpy as np
import torch 
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.structure import get_astr_from_x_tensor
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.models.data_utils import minmax, inv_minmax


class LatentSpaceObjective:
    '''Base class for any latent space optimization task
        class supports any optimization task with accompanying VAE
        such that during optimization, latent space points (z) 
        must be passed through the VAE decoder to obtain 
        original input space points (x) which can then 
        be passed into the oracle to obtain objective values (y)''' 

    def __init__(
        self,
        pool_dict={},
        labels_count=0,
        num_calls=0,
        task_id='',
        nc_vae_params=None,
        scaler_X=None,
        scaler_Y=None,
        energy_code=None,
        pre_trained_matgl_model='MEGNet-MP-2018.6.1-Eform',
        ):

        # Initialize a comparator class with the pool of all existing structures
        self.initialize_comparator()

        # NOTE: pool_dict and label_count are initiated in LOLBO_state, and 
        # will be maintained in sync throughout LSO and NCO
        # dict used to track labels with their xs (input arrays) and 
        # scores (ys) queried during optimization
        self.pool_dict = pool_dict # xs_to_scores_dict 
        
        # track total number of times the oracle has been called
        self.num_calls = num_calls
        
        # string id for optimization task, often used by oracle
        #   to differentiate between similar tasks (ie for guacamol)
        self.task_id = task_id

        self.vae_params = nc_vae_params
        self.scaler_X = scaler_X 
        self.scaler_Y = scaler_Y

        self.energy_code = energy_code

        # load in pretrained VAE, store in variable self.vae
        self.vae = None
        self.initialize_vae()
        self.megnet_short_model = None
        self.initialize_megnet_short_model(pre_trained_matgl_model)
        assert self.vae is not None
        assert self.megnet_short_model is not None


    def __call__(self, z, last_key_idx):
        ''' Input 
                z: a numpy array or pytorch tensor of latent space points
                start_key_idx: The starting integer index for the new set of sampled zs (in latent space). 
                Used to add new samples to pool_di with correct key.
            Output
                out_dict['valid_zs'] = the zs which decoded to valid xs 
                out_dict['decoded_xs'] = an array of valid xs obtained from input zs
                out_dict['scores']: an array of valid scores obtained from input zs
        '''
        # VSCK: TODO: Keep this graph embeds and structure from x_tensor parts in 
        # NanocrystalObjective as it is more specific to the NCs
        if type(z) is np.ndarray: 
            z = torch.from_numpy(z).float()
        decoded_xs = self.vae_decode(z)

        # un-scale the decoded_xs tensor
        descaled_decoded_xs = inv_minmax(decoded_xs.detach().cpu(), self.scaler_X)

        x_keys = [f'sample_{last_key_idx+i+1}' for i in range(decoded_xs.shape[0])]
        scores = []
        astr_xs = []
        failed_inds = []
        for ind, x_descaled in enumerate(descaled_decoded_xs):
            # VSCK: First make sure that the decoded structure is not a duplicate
            # of structures present in the pool
            astr_x = get_astr_from_x_tensor(x_descaled, self.vae_params)
            duplicate = False
            dupe_key = None
            # TODO: Check duplicates with Comparator from FANTASTX
            if dupe_key is not None:
                duplicate = True

            invalid_astr = False
            given_syms = list(self.energy_code.sym_mu_dict.keys())
            curr_syms = list(astr_x.composition.as_dict().keys())
            if not all(sym in given_syms for sym in curr_syms): 
                invalid_astr = True

            if not duplicate and not invalid_astr: # otherwise call the oracle to get score
                # Call VASP or pre-trained model (No need of graph embeddings)
                try:
                    init_astr, rlxd_astr, init_score, rlxd_score = self.query_oracle(
                                                    astr_x, x_keys[ind])
                except:
                    print (f"Failed query_oracle on {x_keys[ind]}")
                    self.num_calls += 1
                    failed_inds.append(ind)
                    continue
            
                scaled_init_score = self.scaler_Y.transform([[init_score]])[0][0]
                scaled_rlxd_score = self.scaler_Y.transform([[rlxd_score]])[0][0]
                scores.append(scaled_init_score)
                scores.append(scaled_rlxd_score)
                astr_xs.append(init_astr)
                astr_xs.append(rlxd_astr)
            else:
                failed_inds.append(ind)

        # convert scores to array
        scores_arr = np.array(scores).astype('float32')

        # Get Xs from the astr_xs
        PCs = self.get_PCs_from_astrs(astr_xs)
        scaled_Xs, _ = minmax(PCs, self.scaler_X)

        # Get graph_embeds from astr_xs
        graph_embeds = []
        for astr_x in astr_xs:
            ge = self.megnet_short_model.predict_structure(astr_x)
            ge = graph_embeds.detach().cpu().numpy()
            graph_embeds.append(ge)
        graph_embeds = np.array(graph_embeds).astype('float32')

        # Get Zs for all the new astrs
        Zs = self.vae.encode(torch.tensor(scaled_Xs), torch.tensor(graph_embeds))


        valid_x_dirs = [x_keys[i] for i in range(len(x_keys)) if i not in failed_inds]
        rename_x_dirs = [f'sample_{last_key_idx+ 2*i+1}' for i in range(len(valid_x_dirs))]
        self.energy_code.rename_dirs(x_keys, valid_x_dirs, rename_x_dirs)
        

        # update pool_dict with new samples (decoded valid_zs)
        # NOTE: VSCK: the pool_dict is in objective is doing bookkeeping.
        # The out_dict is used to update the lolbo_state train_x/y/z on 
        # every iteration. Both are doing the same job. Remove one of them later.
        x_next_keys = [] 
        for i in range(len(scores_arr)):
            key = f'sample_{last_key_idx+i+1}'
            x_next_keys.append(key)
            key_dict = {'x_tensor': scaled_Xs[i], 
                        'astr': astr_xs[i],
                        'graph_embeds': graph_embeds,
                        'score': scores_arr[i]}
            self.pool_dict[key] = key_dict 


        out_dict = {}
        out_dict['scores'] = scores_arr                                # ndarray
        out_dict['valid_zs'] = Zs                                      # tensor
        out_dict['decoded_xs_tensor'] = scaled_Xs                      # tensor
        out_dict['x_next_keys'] = x_next_keys                          # str list
        out_dict['decoded_xs_graph_embeds'] = graph_embeds             # ndarray
        #out_dict['decoded_xs_keys'] = decoded_keys # VSCK: The xs_keys are stored in Lolbo_State; only used in lolbo_state; So not generated in Objective
        return out_dict


    def vae_decode(self, z):
        '''Input
                z: a tensor latent space points
            Output
                a corresponding list of the decoded input space 
                items output by vae decoder 
        '''
        raise NotImplementedError("Must implement vae_decode()")


    def query_oracle(self, x):
        ''' Input: 
                a single input space item x
            Output:
                method queries the oracle and returns 
                the corresponding score y,
                or np.nan in the case that x is an invalid input
        '''
        raise NotImplementedError("Must implement query_oracle() specific to desired optimization task")


    def initialize_vae(self):
        ''' Sets variable self.vae to the desired pretrained vae '''
        raise NotImplementedError("Must implement method initialize_vae() to load in vae for desired optimization task")

    def initialize_comparator(self):
        ''' Sets variable self.comparator to the Comparator class with appropriate parameters 
        Currently using bag-of-bonds with respective thresholds. '''
        raise NotImplementedError("Must implement method initialize_comparator() with bag-of-bonds comparator")

    def initialize_megnet_short_model(self, pre_trained_matgl_model):
        '''    Loads the pre-trained MEGNet model initializes the Short model 
    
        Matgl pre-trained models as of August 2023: 

        'M3GNet-MP-2018.6.1-Eform', 
        'M3GNet-MP-2021.2.8-DIRECT-PES', 
        'M3GNet-MP-2021.2.8-PES', 
        'MEGNet-MP-2018.6.1-Eform', 
        'MEGNet-MP-2019.4.1-BandGap-mfi'  '''
        raise NotImplementedError("Must implement method initialize_megnet_short_model() with one of matgl megnet models")


    def vae_forward(self, xs_batch):
        ''' Input: 
                a list xs 
            Output: 
                z: tensor of resultant latent space codes 
                    obtained by passing the xs through the encoder
                vae_loss: the total loss of a full forward pass
                    of the batch of xs through the vae 
                    (ie reconstruction error)
        '''
        raise NotImplementedError("Must implement method vae_forward() (forward pass of vae)")
