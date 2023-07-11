import numpy as np
import torch 


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
        task_id=''
        ):

        # Initialize a comparator class with the pool of all existing structures
        self.initialize_comparator()

        # NOTE: pool_dict and label_count are initiated in LOLBO_state, and 
        # will be maintained in sync throughout LSO and NCO
        # dict used to track labels with their xs (input arrays) and 
        # scores (ys) queried during optimization
        self.pool_dict = pool_dict # xs_to_scores_dict 
        self.label_count = labels_count
        
        # track total number of times the oracle has been called
        self.num_calls = num_calls
        
        # string id for optimization task, often used by oracle
        #   to differentiate between similar tasks (ie for guacamol)
        self.task_id = task_id

        # load in pretrained VAE, store in variable self.vae
        self.vae = None
        self.initialize_vae()
        assert self.vae is not None


    def __call__(self, z):
        ''' Input 
                z: a numpy array or pytorch tensor of latent space points
            Output
                out_dict['valid_zs'] = the zs which decoded to valid xs 
                out_dict['decoded_xs'] = an array of valid xs obtained from input zs
                out_dict['scores']: an array of valid scores obtained from input zs
        '''
        if type(z) is np.ndarray: 
            z = torch.from_numpy(z).float()
        decoded_xs = self.vae_decode(z)
        scores = []
        for pc_x in decoded_xs:
            # VSCK: First make sure that the decoded structure is not a duplicate
            # of structures present in the pool

            # VSCK: Get label key for pool_dict
            key = 'sample_{}'.format(self.labels_count + 1)
            astr_x = get_astr_from_PC(pc_x)
            
            # Add the new sample to pool_dict

            dupe_key = None
            # TODO: Check duplicates with Comparator from FANTASTX
            if dupe_key is not None:
                score = self.pool_dict[dupe_key]['score']
                print ('Place holder for structure comparator')
            else: # otherwise call the oracle to get score
                score = self.query_oracle(x)
                key_dict = {
                            'astr': astr_x,
                            'PC_x': pc_x,
                            'score': score
                }
                self.pool_dict[key] = key_dict 
                # TODO: Remove this molecule related condition and insert 
                # any other nanocrystal related condition if needed
                # track number of oracle calls 
                #   nan scores happen when we pass an invalid
                #   molecular string and thus avoid calling the
                #   oracle entirely
                if np.logical_not(np.isnan(score)):
                    self.num_calls += 1
                self.labels_count += 1

            scores.append(score)

        scores_arr = np.array(scores)
        decoded_xs = np.array(decoded_xs)
        # get valid zs, xs, and scores
        bool_arr = np.logical_not(np.isnan(scores_arr)) 
        decoded_xs = decoded_xs[bool_arr]
        scores_arr = scores_arr[bool_arr]
        valid_zs = z[bool_arr]

        out_dict = {}
        out_dict['scores'] = scores_arr
        out_dict['valid_zs'] = valid_zs
        out_dict['decoded_xs'] = decoded_xs
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
