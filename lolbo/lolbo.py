import torch
import gpytorch
import math
from gpytorch.mlls import PredictiveLogLikelihood 
from lolbo_nanocrystal.lolbo.utils.bo_utils.turbo import TurboState, update_state, generate_batch
from lolbo_nanocrystal.lolbo.utils.utils import update_models_end_to_end, update_surr_model
from lolbo_nanocrystal.lolbo.utils.bo_utils.ppgpr import GPModelDKL
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.structure import get_astr_from_x_tensor


class LOLBOState:

    def __init__(
        self,
        objective,
        train_x_keys,
        train_x_tensor,
        graph_embeds,
        train_y,
        train_z,
        k=1_000,
        minimize=False,
        num_update_epochs=2,
        init_n_epochs=20,
        learning_rte=0.01,
        bsz=10,
        acq_func='ts',
        verbose=True,
        ):
        # NOTE: Included graph embeddings to be complete. However, it is not needed for LOLBO
        # train_x (Structure, input PC) and train_y (score) is enough. Remove graph_embeds later..

        self.objective          = objective         # objective with vae for particular task
        self.train_x_keys       = train_x_keys
        self.train_x_tensor     = train_x_tensor    # initial train x data
        self.graph_embeds       = graph_embeds      # MEGNet graph embeddings to use in NanoCrystalVAE
        self.train_y            = train_y           # initial train y data
        self.train_z            = train_z           # initial train z data
        self.minimize           = minimize          # if True we want to minimize the objective, otherwise we assume we want to maximize the objective
        self.k                  = k                 # track and update on top k scoring points found
        self.num_update_epochs  = num_update_epochs # num epochs update models
        self.init_n_epochs      = init_n_epochs     # num epochs train surr model on initial data
        self.learning_rte       = learning_rte      # lr to use for model updates
        self.bsz                = bsz               # acquisition batch size
        self.acq_func           = acq_func          # acquisition function (Expected Improvement (ei) or Thompson Sampling (ts))
        self.verbose            = verbose

        assert acq_func in ["ei", "ts"]
        if minimize:
            self.train_y = self.train_y * -1

        self.progress_fails_since_last_e2e = 0
        self.tot_num_e2e_updates = 0
        self.best_score_seen = torch.max(train_y)
        self.best_x_seen = train_x_keys[torch.argmax(train_y.squeeze())]
        self.initial_model_training_complete = False # initial training of surrogate model uses all data for more epochs
        self.new_best_found = False

        self.initialize_top_k()
        self.initialize_surrogate_model()
        self.initialize_tr_state()
        #self.initialize_xs_to_scores_dict() 
        # VSCK: Renaming xs_to_scores_dict to pool_dict
        # NOTE: Although pool_dict is actually created here, it is kept as objective attribute. So, naming appropriately
        self.initialize_pool_dict_to_objective()


    #def initialize_xs_to_scores_dict(self,):
    #    # put initial xs and ys in dict to be tracked by objective
    #    init_xs_to_scores_dict = {}
    #    for idx, x in enumerate(self.train_x):
    #        init_xs_to_scores_dict[x] = self.train_y.squeeze()[idx].item()
    #    self.objective.xs_to_scores_dict = init_xs_to_scores_dict

    def initialize_pool_dict_to_objective(self):
        ''' Generate a label for each sample; 
        Assign astr (pymatgen structure), PC array (xs), and score (ys) for each label'''
        init_pool_dict = {}
        for idx, key in enumerate(self.train_x_keys):
            x_tensor = self.train_x_tensor[idx]
            astr_x = get_astr_from_x_tensor(x_tensor)
            graph_embeds_x = self.graph_embeds[idx]
            score = self.train_y.squeeze()[idx].item()
            init_pool_dict[key] = {
                'x_tensor': x_tensor,
                'astr': astr_x,
                'graph_embeds': graph_embeds_x,
                'score': score
            }
        self.objective.pool_dict = init_pool_dict


    def initialize_top_k(self):
        ''' Initialize top k x, y, and zs'''
        # track top k scores found
        self.top_k_scores, top_k_idxs = torch.topk(self.train_y.squeeze(), min(self.k, len(self.train_y)))
        self.top_k_scores = self.top_k_scores.tolist()
        top_k_idxs = top_k_idxs.tolist()
        self.top_k_xs_keys = [self.train_x_keys[i] for i in top_k_idxs]
        self.top_k_xs_tensor = [self.train_x_tensor[i] for i in top_k_idxs]
        self.top_k_zs = [self.train_z[i].unsqueeze(-2) for i in top_k_idxs]
        self.top_k_graph_embeds = [self.graph_embeds[i] for i in top_k_idxs]


    def initialize_tr_state(self):
        # initialize turbo trust region state
        self.tr_state = TurboState( # initialize turbo state
            dim=self.train_z.shape[-1],
            batch_size=self.bsz, 
            best_value=torch.max(self.train_y).item()
            )

        return self


    def initialize_surrogate_model(self ):
        likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda() 
        n_pts = min(self.train_z.shape[0], 1024)
        self.model = GPModelDKL(self.train_z[:n_pts, :].cuda(), likelihood=likelihood ).cuda()
        self.mll = PredictiveLogLikelihood(self.model.likelihood, self.model, num_data=self.train_z.size(-2))
        self.model = self.model.eval() 
        self.model = self.model.cuda()

        return self


    def update_next(self, z_next_, y_next_, x_next_tensor, x_next_keys, graph_embeds_next_, acquisition=False):
        '''Add new points (z_next, y_next, x_next) to train data
            and update progress (top k scores found so far)
            and update trust region state
        '''
        z_next_ = z_next_.detach().cpu() 
        y_next_ = y_next_.detach().cpu()
        x_next_tensor = x_next_tensor.detach().cpu()
        graph_embeds_next_ = graph_embeds_next_.detach().cpu()

        if len(y_next_.shape) > 1:
            y_next_ = y_next_.squeeze() 
        if len(z_next_.shape) == 1:
            z_next_ = z_next_.unsqueeze(0)
        progress = False
        for i, score in enumerate(y_next_):
            self.train_x_keys.append(x_next_keys[i])
            torch.cat((self.train_x_tensor, x_next_tensor[i].unsqueeze(0)), dim=0)
            if len(self.top_k_scores) < self.k: 
                # if we don't yet have k top scores, add it to the list
                self.top_k_scores.append(score.item())
                self.top_k_xs_keys.append(x_next_keys[i])
                self.top_k_xs_tensor.append(x_next_tensor[i])
                self.top_k_zs.append(z_next_[i].unsqueeze(-2))
                self.top_k_graph_embeds.append(graph_embeds_next_[i])
            elif score.item() > min(self.top_k_scores) and (x_next_keys[i] not in self.top_k_xs_keys):
                # if the score is better than the worst score in the top k list, upate the list
                min_score = min(self.top_k_scores)
                min_idx = self.top_k_scores.index(min_score)
                self.top_k_scores[min_idx] = score.item()
                self.top_k_xs_keys[min_idx] = x_next_keys[i]
                self.top_k_xs_tensor[min_idx] = x_next_tensor[i]
                self.top_k_zs[min_idx] = z_next_[i].unsqueeze(-2) # .cuda()
                self.top_k_graph_embeds[min_idx] = graph_embeds_next_[i]
            #if we imporve
            if score.item() > self.best_score_seen:
                self.progress_fails_since_last_e2e = 0
                progress = True
                self.best_score_seen = score.item() #update best
                self.best_x_seen = x_next_keys[i]
                self.new_best_found = True
        if (not progress) and acquisition: # if no progress msde, increment progress fails
            self.progress_fails_since_last_e2e += 1

        if acquisition:
            self.tr_state = update_state(state=self.tr_state, Y_next=y_next_)
        self.train_z = torch.cat((self.train_z, z_next_), dim=0)
        self.train_y = torch.cat((self.train_y, y_next_), dim=0)
        self.graph_embeds = torch.cat((self.graph_embeds, graph_embeds_next_), dim=0)

        return self


    def update_surrogate_model(self): 
        if not self.initial_model_training_complete:
            # first time training surr model --> train on all data
            n_epochs = self.init_n_epochs
            train_z = self.train_z
            train_y = self.train_y.squeeze(-1)
        else:
            # otherwise, only train on most recent batch of data
            n_epochs = self.num_update_epochs
            train_z = self.train_z[-self.bsz:]
            train_y = self.train_y[-self.bsz:].squeeze(-1)
            
        self.model = update_surr_model(
            self.model,
            self.mll,
            self.learning_rte,
            train_z,
            train_y,
            n_epochs
        )
        self.initial_model_training_complete = True

        return self


    def update_models_e2e(self):
        '''Finetune VAE end to end with surrogate model'''
        self.progress_fails_since_last_e2e = 0
        new_x_keys = self.train_x_keys[-self.bsz:] 
        train_x_keys = new_x_keys + self.top_k_xs_keys
        # Use new x_keys to get the other data
        train_x_tensors, train_graph_embeds, train_y_tensors = [], [], []
        for each_key in train_x_keys:
            train_x_tensors.append(self.objective.pool_dict[each_key]['x_tensor'])
            train_graph_embeds.append(self.objective.pool_dict[each_key]['graph_embeds'])
            train_y_tensors.append(self.objective.pool_dict[each_key]['score'])

        # convert lists to tensors
        train_x_tensors = torch.tensor(train_x_tensors, dtype=torch.float)
        train_graph_embeds = torch.tensor(train_graph_embeds, dtype=torch.float)
        train_y_tensors = torch.tensor(train_y_tensors, dtype=torch.float)

        self.objective, self.model = update_models_end_to_end(
            train_x_tensors,
            train_graph_embeds,
            train_y_tensors,
            self.objective,
            self.model,
            self.mll,
            self.learning_rte,
            self.num_update_epochs
        )
        self.tot_num_e2e_updates += 1

        return self


    def recenter(self):
        '''Pass SELFIES strings back through
            VAE to find new locations in the
            new fine-tuned latent space
        '''
        self.objective.vae.eval()
        self.model.train()
        optimizer1 = torch.optim.Adam([{'params': self.model.parameters(),'lr': self.learning_rte} ], lr=self.learning_rte)
        new_xs = self.train_x_tensor[-self.bsz:]
        new_graph_embeds = self.graph_embeds[-self.bsz:]
        train_x = torch.cat((new_xs, self.top_k_x_tensors), dim=0)
        train_graph_embeds = torch.cat((new_graph_embeds, self.top_k_graph_embeds), dim=0)
        bsz = self.bsz
        num_batches = math.ceil(train_x.shape[0] / bsz) 
        for _ in range(self.num_update_epochs):
            for batch_ix in range(num_batches):
                start_idx, stop_idx = batch_ix*bsz, (batch_ix+1)*bsz
                batch_x_tensor = train_x[start_idx:stop_idx] 
                batch_graph_embeds = train_graph_embeds[start_idx:stop_idx]
                z, _ = self.objective.vae_forward(batch_x_tensor, batch_graph_embeds) # VSCK: TODO: Add graph_embeds to Lolbo_State attributes
                out_dict = self.objective(z, start_key_idx=len(self.train_x_keys))
                scores_arr = out_dict['scores'] 
                valid_zs = out_dict['valid_zs']
                decoded_xs_tensor = out_dict['decoded_xs_tensor']
                #labels_list = out_dict['labels_list'] # VSCK: TODO: Check the out_dict from latent_space_objective.__call__()
                if len(scores_arr) > 0: # if some valid scores
                    scores_arr = torch.from_numpy(scores_arr)
                    if self.minimize:
                        scores_arr = scores_arr * -1
                    pred = self.model(valid_zs)
                    loss = -self.mll(pred, scores_arr.cuda())
                    optimizer1.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer1.step() 
                    with torch.no_grad(): 
                        z = z.detach().cpu()
                        self.update_next(z, scores_arr, decoded_xs_tensor)
            torch.cuda.empty_cache()
        self.model.eval() 

        return self


    def acquisition(self):
        '''Generate new candidate points, 
        evaluate them, and update data
        '''
        # 1. Generate a batch of candidates in 
        #   trust region using surrogate model
        z_next = generate_batch(
            state=self.tr_state,
            model=self.model,
            X=self.train_z,
            Y=self.train_y,
            batch_size=self.bsz, 
            acqf=self.acq_func,
        )
        # 2. Evaluate the batch of candidates by calling oracle
        with torch.no_grad():
            out_dict = self.objective(z_next, last_key_idx=len(self.train_x_keys))
            z_next = out_dict['valid_zs']
            y_next = out_dict['scores']
            x_next_tensor = out_dict['decoded_xs_tensor']
            x_next_keys = out_dict['x_next_keys']
            graph_embeds_next = out_dict['decoded_xs_graph_embeds']
            if self.minimize:
                y_next = y_next * -1

        # 3. Add new evaluated points to dataset (update_next)
        if len(y_next) != 0:            
            y_next = torch.from_numpy(y_next).float()
            x_next_tensor = torch.from_numpy(x_next_tensor).float()
            graph_embeds_next = torch.from_numpy(graph_embeds_next).float()
            self.update_next(
                z_next,
                y_next,
                x_next_tensor,
                x_next_keys,
                graph_embeds_next,
                acquisition=True
            )
        else:
            self.progress_fails_since_last_e2e += 1
            if self.verbose:
                print("GOT NO VALID Y_NEXT TO UPDATE DATA, RERUNNING ACQUISITOIN...")
