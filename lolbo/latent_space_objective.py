import numpy as np
import torch 
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.structure import get_astr_from_x_tensor, get_graph_embeds_from_astr


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
        energy_code=None,
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

        self.energy_code = energy_code

        # load in pretrained VAE, store in variable self.vae
        self.vae = None
        self.initialize_vae()
        assert self.vae is not None


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
        x_keys = [f'sample_{last_key_idx+i+1}' for i in range(decoded_xs.shape[0])]
        scores = []
        astr_xs = []
        for ind, x_tensor in enumerate(decoded_xs):
            # VSCK: First make sure that the decoded structure is not a duplicate
            # of structures present in the pool
            astr_x = get_astr_from_x_tensor(x_tensor)
            dupe_key = None
            # TODO: Check duplicates with Comparator from FANTASTX
            if dupe_key is not None:
                score = self.pool_dict[dupe_key]['score']
                print ('Place holder for structure comparator')

            else: # otherwise call the oracle to get score
                # Call VASP or pre-trained model (No need of graph embeddings)
                score = self.query_oracle(astr_x, x_keys[ind])
                if np.logical_not(np.isnan(score)):
                    self.num_calls += 1

            scores.append(score)
            astr_xs.append(astr_x)

        scores_arr = np.array(scores).astype('float32')
        decoded_xs = np.array(decoded_xs.detach().cpu()).astype('float32')
        # get valid zs, xs, and scores
        bool_arr = np.logical_not(np.isnan(scores_arr)) 
        decoded_xs = decoded_xs[bool_arr]
        scores_arr = scores_arr[bool_arr]
        valid_zs = z[bool_arr]

        valid_x_keys = x_keys[bool_arr]
        new_x_keys = [f'sample_{last_key_idx+i+1}' for i in range(len(valid_x_keys))]
        self.energy_code.rename_dirs(x_keys, valid_x_keys, new_x_keys)
        
        # update pool_dict with new samples (decoded valid_zs)
        # NOTE: VSCK: the pool_dict is in objective is doing bookkeeping.
        # The out_dict is used to update the lolbo_state train_x/y/z on 
        # every iteration. Both are doing the same job. Remove one of them later.
        x_next_keys = [] 
        decoded_xs_graph_embeds = []
        for i in range(len(valid_zs)):
            key = f'sample_{last_key_idx+i+1}'
            x_next_keys.append(key)
            graph_embeds_x = get_graph_embeds_from_astr(astr_xs[i])
            decoded_xs_graph_embeds.append(graph_embeds_x)
            key_dict = {'x_tensor': decoded_xs[i], 
                        'astr': astr_xs[i],
                        'graph_embeds': graph_embeds_x,
                        'score': scores_arr[i]}
            self.pool_dict[key] = key_dict 

        decoded_xs_graph_embeds = np.array(decoded_xs_graph_embeds).astype('float32')

        out_dict = {}
        out_dict['scores'] = scores_arr
        out_dict['valid_zs'] = valid_zs
        out_dict['decoded_xs_tensor'] = decoded_xs
        out_dict['x_next_keys'] = x_next_keys
        out_dict['decoded_xs_graph_embeds'] = decoded_xs_graph_embeds
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
