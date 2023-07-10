import os 
from math import log
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, LearningRateMonitor, EarlyStopping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from typing import List, Callable, Union, Any, TypeVar, Tuple
# Tensor = TypeVar('torch.tensor')
from abc import abstractmethod
from operator import itemgetter

import joblib, json

from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.models.data_utils import minmax, inv_minmax, MAE, MAPE, MAE_site_coor
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.models.data import IrOx_Dataset, IrOxDataModule

from pytorch_lightning import seed_everything
seed_everything(199, workers=True)


BATCH_SIZE = 256
ENCODER_LR = 1e-4
DECODER_LR = 1e-4
ENCODER_WARMUP_STEPS = 100
DECODER_WARMUP_STEPS = 100
AGGRESSIVE_STEPS = 5
TORCH_DISTRIBUTED_DEBUG = 'INFO'



class EncoderAndRegressor(nn.Module):
    def __init__(
            self, 
            input_dim: int = 164,
            channel_dim: int = 3, 
            regression_dim: int = 1,
            graph_embds_dim: int = 16,
            coeffs: Tuple = (1, 2, 10,),
            latent_dim: int = 32, 
            max_filters: int = 128, 
            filter_size: List = [5, 3, 3], 
            strides: List = [2, 2, 1],
    ) -> None:
        super(EncoderAndRegressor, self).__init__()

        # Save attributes
        self.input_dim = input_dim
        self.channel_dim = channel_dim
        self.regression_dim = regression_dim
        self.graph_embds_dim = graph_embds_dim
        self.latent_dim = latent_dim
        self.max_filters = max_filters
        self.filter_size = filter_size
        self.strides = strides
        self.coeff_recon = coeffs[0]
        self.coeff_KL = coeffs[1]
        self.coeff_reg = coeffs[2]

        # Build Encoder layers
        modules = []

        in_channels = [channel_dim, max_filters//4, max_filters//2]
        out_channels = [max_filters//4, max_filters//2, max_filters]
        paddings = [2, 1, 1]

        for i, stride in enumerate(self.strides):
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels = in_channels[i], 
                            out_channels = out_channels[i],
                            kernel_size = self.filter_size[i], 
                            stride = stride, 
                            padding = paddings[i]),
                    nn.BatchNorm1d(out_channels[i]),
                    nn.LeakyReLU(0.2))
            )
        self.encoder_conv_layers = nn.Sequential(*modules)

        # final encoder layers if latent space < 64
        self.encoder_final_2 = nn.Sequential(
            nn.Linear(self.input_dim//4 * out_channels[-1] + graph_embds_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.Sigmoid()
        ) 

        # fc_mu and fc_var layer if latent_space < 64
        self.fc_mu_2 = nn.Linear(256, latent_dim)
        self.fc_var_2 = nn.Linear(256, latent_dim)

        # if latent_dim < 64:
        self.reg_layers_2 = nn.Sequential(
            nn.Linear(latent_dim, regression_dim),
            nn.Sigmoid()
        )

    def forward(self, input: Tensor, graph_embeds: Tensor) -> List[Tensor]:
        input = torch.swapaxes(input, 1, 2)
        result = self.encoder_conv_layers(input)
        result = torch.flatten(result, start_dim=1)

        # concatenate the respective graph embedds to each flattened tensor
        result = torch.cat((result, graph_embeds), dim=1)
        result = self.encoder_final_2(result)
        mu = self.fc_mu_2(result)
        log_var = self.fc_var_2(result)

        reg_res = self.reg_layers_2(mu)

        return [mu, log_var, reg_res]
    

class Decoder(nn.Module):
    def __init__(
            self, 
            input_dim: int = 164,
            channel_dim: int = 3, 
            regression_dim: int = 1,
            graph_embds_dim: int = 16,
            coeffs: Tuple = (1, 2, 10,),
            latent_dim: int = 32, 
            max_filters: int = 128, 
            filter_size: List = [5, 3, 3], 
            strides: List = [2, 2, 1],            
    ):
        super(Decoder, self).__init__()

        # Save attributes
        self.input_dim = input_dim
        self.channel_dim = channel_dim
        self.regression_dim = regression_dim
        self.graph_embds_dim = graph_embds_dim
        self.latent_dim = latent_dim
        self.max_filters = max_filters
        self.filter_size = filter_size
        self.strides = strides

        # Build Decoder
        # decoder initial layers if latent dim < 64
        self.decoder_initial_2 = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, max_filters * self.input_dim//4),
            nn.ReLU()
        )

        # BatchNorm for decoder
        self.decoder_batch_norm_1 = nn.BatchNorm2d(max_filters)

        dec_strides = strides.copy()
        dec_strides.reverse()
        dec_filter_size = filter_size.copy()
        dec_filter_size.reverse()

        dec_in_channels = [max_filters, max_filters//2, max_filters//4]
        dec_out_channels = [max_filters//2, max_filters//4, channel_dim]

        paddings = [1, 1, 2]
        out_paddings = [0, 1, 1]

        # decoder conv transpose layers
        modules = []
        for i in range(len(dec_strides)-1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=dec_in_channels[i],
                        out_channels=dec_out_channels[i],
                        kernel_size=(dec_filter_size[i], 1),
                        stride=(dec_strides[i], 1),
                        padding=(paddings[i], 0),
                        output_padding=(out_paddings[i], 0)),
                    nn.BatchNorm2d(dec_out_channels[i]),
                    nn.ReLU())
            )

        self.decoder_conv_transpose_layers = nn.Sequential(*modules)

        self.decoder_final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dec_in_channels[-1],
                               out_channels=dec_out_channels[-1],
                               kernel_size=(dec_filter_size[-1], 1),
                               stride=(dec_strides[-1], 1),
                               padding=(paddings[-1], 0),
                               output_padding=(out_paddings[-1], 0)),
            nn.Sigmoid()
        )

    def forward(self, z: Tensor) -> Any:
        
        result = self.decoder_initial_2(z)
        result = result.view(-1, self.max_filters, self.input_dim//4, 1)
        result = self.decoder_batch_norm_1(result)
        result = self.decoder_conv_transpose_layers(result)
        result = self.decoder_final_layer(result)
        result = result.view(-1, self.channel_dim, self.input_dim)
        result = torch.swapaxes(result, 1, 2)

        return result


class NanoCrystalVAE(pl.LightningModule):
    def __init__(
            self, 
            input_dim: int = 164,
            channel_dim: int = 3, 
            regression_dim: int = 1,
            graph_embds_dim: int = 16,
            coeffs: Tuple = (1, 2, 10,),
            latent_dim: int = 32, 
            max_filters: int = 128, 
            filter_size: List = [5, 3, 3], 
            strides: List = [2, 2, 1],       
    ) -> None:
        super(NanoCrystalVAE, self).__init__()
        self.save_hyperparameters()

        self.encoder = EncoderAndRegressor(**self.hparams)
        self.decoder = Decoder(**self.hparams)

        self.validation_step_outputs = []
        self.training_step_outputs = []


    @abstractmethod
    def loss_function(*args: Any, **kwargs) -> Tensor:
        recon_outputs = args[0]
        encoder_inputs = args[1]
        mu = args[2]
        log_var = args[3]
        reg_outputs = args[4]
        regression_inputs = args[5]

        coeff_recon = kwargs['coeff_recon']
        coeff_KL = kwargs['coeff_KL']
        coeff_reg = kwargs['coeff_reg']

        recons_loss = torch.sum(torch.square(encoder_inputs - recon_outputs))
        kld_loss = -0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)
        reg_loss = torch.sum(torch.square(regression_inputs - reg_outputs))
        
        loss = torch.mean(coeff_recon * recons_loss + coeff_KL * kld_loss + coeff_reg * reg_loss)
        return {'loss': loss, 'Reconstruction_loss':recons_loss, 
                'KLD_loss':-kld_loss, 'Regression_loss': reg_loss}

    # This is same as sample_posterior in lolbo
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    # This is similar to sample_prior in lolbo
    def sample(self, num_samples:int) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        #z = z.to(current_device)

        samples = self.decode(z)
        return samples
    
    def forward(self, inputs: Tensor, graph_embeds: Tensor):
        mu_, log_var_, reg_output_ = self.encoder(input=inputs, graph_embeds=graph_embeds)
        z = self.reparameterize(mu_, log_var_)
        reconstructed_output_ = self.decoder(z)

        return dict(
            mu=mu_,
            log_var=log_var_,
            reg_output=reg_output_,
            reconstructed_output=reconstructed_output_
        )

    def training_step(self, batch, batch_idx):
        inputs, target_ys = batch
        input_xs, embed_xs = inputs
        outputs_dict = self(input_xs, embed_xs)

        mu_, log_var_, reg_output_, reconstructed_output_ = itemgetter(
            'mu', 'log_var', 'reg_output', 'reconstructed_output')(outputs_dict)
        
        # Compute the loss and its gradients
        loss_fn_args = [reconstructed_output_, input_xs, mu_, log_var_, reg_output_, target_ys]
        loss_fn_kwargs = {
            'coeff_recon': self.hparams.coeffs[0],
            'coeff_KL': self.hparams.coeffs[1], 
            'coeff_reg': self.hparams.coeffs[2]
        }
        loss_dict = NanoCrystalVAE.loss_function(*loss_fn_args, **loss_fn_kwargs)

        # Log losses every step
        for k, v in loss_dict.items():
            self.log('train/' + k, v, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        inputs, target_ys = batch
        input_xs, embed_xs = inputs
        outputs_dict = self(input_xs, embed_xs)

        #mu, log_var, reg_output, reconstructed_output = outputs
        mu_, log_var_, reg_output_, reconstructed_output_ = itemgetter(
            'mu', 'log_var', 'reg_output', 'reconstructed_output')(outputs_dict)
        
        # Compute the loss and its gradients
        loss_fn_args = [reconstructed_output_, input_xs, mu_, log_var_, reg_output_, target_ys]
        loss_fn_kwargs = {
            'coeff_recon': self.hparams.coeffs[0],
            'coeff_KL': self.hparams.coeffs[1], 
            'coeff_reg': self.hparams.coeffs[2]
        }
        loss_dict = NanoCrystalVAE.loss_function(*loss_fn_args, **loss_fn_kwargs)

        # Log losses every step
        for k, v in loss_dict.items():
            self.log('validation/' + k, v, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        return loss_dict['loss']    

    def configure_optimizers(self):

        encoder_params = []
        decoder_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'encoder' in name:
                    encoder_params.append(param)
                elif 'decoder' in name:
                    decoder_params.append(param)
                else:
                    raise ValueError(f'Unknown parameter {name}')

        def encoder_lr_sched(step):
            # Use Linear warmup
            if step < ENCODER_WARMUP_STEPS:
                return min(step / ENCODER_WARMUP_STEPS, 1.)
            else:
                if step >= 250*35:
                    if step % 50*35 == 0:
                        return 0.1
                    else:
                        return 1.
                else:
                    return 1.
                

        def decoder_lr_sched(step):
            if step < ENCODER_WARMUP_STEPS:
                return 0.
            else:
                if step >= 250*35:
                    if step % 50*35 == 0:
                        return 0.1
                    else:
                        return 1.
                if (step - ENCODER_WARMUP_STEPS + 1) % AGGRESSIVE_STEPS == 0:
                    return min((step - ENCODER_WARMUP_STEPS) / (DECODER_WARMUP_STEPS * AGGRESSIVE_STEPS), 1.)
                else:
                    return 0.


        optimizer = RMSprop([
            dict(
                params=encoder_params,
                lr=ENCODER_LR
            ),
            dict(
                params=decoder_params,
                lr=DECODER_LR
            )
        ])

        lr_scheduler = LambdaLR(optimizer, [encoder_lr_sched, decoder_lr_sched])

        return [optimizer], [dict(scheduler=lr_scheduler, interval='step', frequency=1)]



def fit(coeffs=(1, 2, 10,), max_epochs=250, batch_size=64,  
        print_metrics=True, checkpath=None):
    
    ########### 1. Dataset Module Initialization ##############

    path_prefix = '/home/vkolluru/GenerativeModeling/Datasets_164x3'
    inp_arr_path = path_prefix + '/L1_Xs.npy'
    y_vals_path = path_prefix + '/L1_Ys.npy'
    embds_path = path_prefix + '/L1_grph_embds.npy'
    training_dataset = IrOx_Dataset(inp_arr_path, y_vals_path, embds_path, 
                                    transform_xs=True, transform_ys=True,
                                    swap_input_axes=False)

    inp_arr_path = path_prefix + '/Testset_Xs.npy'
    y_vals_path = path_prefix + '/Testset_Ys.npy'
    embds_path = path_prefix + '/Testset_grph_embds.npy'
    validation_dataset = IrOx_Dataset(inp_arr_path, y_vals_path, embds_path, 
                                    transform_xs=True, transform_ys=True,
                                    swap_input_axes=False)

    datamodule = IrOxDataModule(batch_size=batch_size, 
                                train_dataset=training_dataset, 
                                validation_dataset=validation_dataset)
    
    ########### 2. VAEModule initialize
    
    lit_model = NanoCrystalVAE(coeffs=coeffs)

    logger = pl.loggers.TensorBoardLogger(save_dir="lightning_logs", 
                                          name=type(lit_model).__name__)

    check = ModelCheckpoint(
        every_n_epochs=40,
        save_top_k=-1,
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stop_callback = EarlyStopping(
        monitor='validation/loss',
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode='min'
    )

    trainer = pl.Trainer(
        #auto_lr_find=True,
        #fast_dev_run=50,
        gpus=1,
        max_epochs=max_epochs,
        strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=False), # set True when unused layers in __init__
        logger=logger,
        callbacks=[check, lr_monitor, early_stop_callback], #, RichProgressBar()],
        gradient_clip_val=1.,
        gradient_clip_algorithm='norm',
        detect_anomaly=True,
        #deterministic=True, # ensures reproducibility but slow (Non-random methods used)
        #sync_batchnorm=True,
        log_every_n_steps=25,
        enable_model_summary=True,
    )

    trainer.fit(lit_model, datamodule=datamodule, ckpt_path=checkpath)

    print (list(trainer.strategy._lightning_optimizers.values()))

    optimizers = trainer.optimizers
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

    print ("test done")

    if print_metrics:
        if trainer.is_global_zero:
            # Pass the test dataset through the trained VAE
            testset_loader = DataLoader(validation_dataset, 
                                        batch_size=len(validation_dataset.input_array), 
                                        shuffle=False)

            lit_model.eval()
            with torch.no_grad():
                for i, batch in enumerate(testset_loader):
                    inputs, target_ys = batch
                    input_xs, embed_xs = inputs
                    outputs = lit_model(input_xs, embed_xs)

                    #mu, log_var, reg_output, reconstructed_output = outputs
                    outputs_dict = outputs
                    reg_output, reconstructed_output = itemgetter(
                        'reg_output', 'reconstructed_output')(outputs_dict)
                
            scaler_x = validation_dataset.input_scaler
            scaler_y = validation_dataset.target_scaler
            max_sites = 29
            val_Nsites = '/home/vkolluru/GenerativeModeling/New_IrO2_datasets/combined_dataset_wo_duplicates/str_reps/DS_TestSet_Nsites.npy'
            Nsites_val = np.load(val_Nsites, allow_pickle=True)
            # Get the elements_string list (from FTCP src)
            ftcp_src_path = '/home/vkolluru/GenerativeModeling/FTCPcode/src'
            elm_str = joblib.load(ftcp_src_path + '/data/element.pkl')

            # Rescale reconstructed output to original values
            X_test_recon_ = inv_minmax(reconstructed_output, scaler_x)
            X_test_recon_[X_test_recon_ < 0.1] = 0
            X_test_ = inv_minmax(validation_dataset.input_array, scaler_x)

            # Get lattice constants, abc # The line with abc
            abc = X_test_[:, len(elm_str), :3]
            abc_recon = X_test_recon_[:, len(elm_str), :3]
            abc_mape = MAPE(abc,abc_recon)

            # Get lattice angles, alpha, beta, and gamma # Line next to abc
            ang = X_test_[:, len(elm_str)+1, :3]
            ang_recon = X_test_recon_[:, len(elm_str)+1, :3]
            ang_mape = MAPE(ang, ang_recon)

            # Get site coordinates
            coor = X_test_[:, len(elm_str)+2:len(elm_str)+2+max_sites, :3]
            coor_recon = X_test_recon_[:, len(elm_str)+2:len(elm_str)+2+max_sites, :3]
            sites_mae = MAE_site_coor(coor, coor_recon, Nsites_val)

            # Get target-learning branch regression error
            prop = 'formation_energy'
            #y_test_hat = regression.predict([X_test, graph_embds_Test], verbose=1)
            y_test_ = scaler_y.inverse_transform(validation_dataset.y_values_array)
            y_test_hat_ = scaler_y.inverse_transform(reg_output)
            reg_mae = MAE(y_test_, y_test_hat_)[0]

            print ('abc MAPE: {}\nang_MAPE: {}\ncoords_MAE: {}\nEf_MAE: {}'.format(
                                        abc_mape, ang_mape, sites_mae, reg_mae))


if __name__ == '__main__':
    fit(coeffs=(1, 2, 15,), max_epochs=500, print_metrics=True)

