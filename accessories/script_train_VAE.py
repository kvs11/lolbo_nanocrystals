"""
A script used to pre-train a NC-VAE. Essentially, a wrapper on NanoCrystalVAE 
from IrOx_VAE.py module. This is a more generalized version that includes 
all variables in config.
"""
#import os 
#from math import log
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, \
                                    LearningRateMonitor, EarlyStopping

import torch
#from torch import Tensor, nn
#from torch.nn import functional as F
from torch.utils.data import DataLoader

#from torch.optim import Adam, RMSprop
#from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR

import numpy as np
#from sklearn.preprocessing import MinMaxScaler

from typing import List, Callable, Union, Any, TypeVar, Tuple
# Tensor = TypeVar('torch.tensor')
#from abc import abstractmethod
from operator import itemgetter

import joblib, json

from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.models.IrOx_VAE \
                    import NanoCrystalVAE
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.models.data_utils \
                    import minmax, inv_minmax, MAE, MAPE, MAE_site_coor
from lolbo_nanocrystal.lolbo.utils.nanocrystal_utils.models.data \
                    import IrOx_Dataset, IrOxDataModule

#from pytorch_lightning import seed_everything
pl.seed_everything(199, workers=True)

def fit(
        gpus: int = 2,
        max_epochs: int = 500,
        batch_size: int = 64,
        print_metrics: bool = True,
        checkpath: str = None,
        log_every_n_steps: int = 25,
        enable_model_summary: bool = True,

        train_PC_array_path: str = "",
        train_Y_array_path: str = "",
        train_embds_path: str = "",
        test_PC_array_path: str = "",
        test_Y_array_path: str = "",
        test_embds_path: str = "",
        test_Nsites_path: str = "",
        max_sites: int = None,
        zero_pad_rows: int = 0,

        # NC-VAE params
        input_dim: int = 145,
        channel_dim: int = 3,
        regression_dim: int = 1,
        graph_embds_dim: int = 16,
        latent_dim: int = 32,
        max_filters: int = 128,
        num_filters: List = None,
        filter_size: Union[List, Tuple] = [5, 3, 3],
        strides: Union[List, Tuple] = [2, 2, 1],
        coeffs: Union[List, Tuple] = (1, 2, 15),
):
    
    ########### 1. Dataset Module Initialization ##############
    training_dataset = IrOx_Dataset(train_PC_array_path, train_Y_array_path, 
                                    train_embds_path, transform_xs=True, 
                                    transform_ys=True, swap_input_axes=False)
    
    validation_dataset = IrOx_Dataset(test_PC_array_path, test_Y_array_path, 
                                    test_embds_path, transform_xs=True, 
                                    transform_ys=True, swap_input_axes=False)

    datamodule = IrOxDataModule(batch_size=batch_size, 
                                train_dataset=training_dataset, 
                                validation_dataset=validation_dataset)
    
    ########### 2. VAEModule initialize
    
    lit_model = NanoCrystalVAE(
        input_dim=input_dim,
        channel_dim=channel_dim,
        regression_dim=regression_dim,
        graph_embds_dim=graph_embds_dim,
        coeffs=coeffs,
        latent_dim=latent_dim,
        max_filters=max_filters,
        num_filters=num_filters,
        filter_size=filter_size,
        strides=strides,
    )

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
        gpus=gpus,
        max_epochs=max_epochs,
        strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=False), # set True when unused layers in __init__
        logger=logger,
        callbacks=[check, lr_monitor, early_stop_callback], #, RichProgressBar()],
        gradient_clip_val=1.,
        gradient_clip_algorithm='norm',
        detect_anomaly=True,
        #deterministic=True, # ensures reproducibility but slow (Non-random methods used)
        #sync_batchnorm=True,
        log_every_n_steps=log_every_n_steps,
        enable_model_summary=enable_model_summary,
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
            Nsites_val = np.load(test_Nsites_path, allow_pickle=True)
            # Get the elements_string list (from FTCP src)
            ftcp_src_path = '/home/vkolluru/GenerativeModeling/FTCPcode/src'
            elm_str = joblib.load(ftcp_src_path + '/data/element.pkl')

            # Rescale reconstructed output to original values
            X_test_recon_ = inv_minmax(reconstructed_output, scaler_x)
            X_test_recon_[X_test_recon_ < 0.1] = 0
            X_test_ = inv_minmax(validation_dataset.input_array, scaler_x)

            # Remove zero_padding at top and bottom if any
            if zero_pad_rows > 0:
                if zero_pad_rows%2 == 0:
                    top_pad = bot_pad = zero_pad_rows / 2
                if zero_pad_rows%2 == 1:
                    top_pad = int(zero_pad_rows/2)
                    bot_pad = top_pad + 1
                X_test_ = X_test_[:, top_pad:-bot_pad]
                X_test_recon_ = X_test_recon_[:, top_pad:-bot_pad]

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
            #y_test_hat = regression.predict([X_test, graph_embds_Test], verbose=1)
            y_test_ = scaler_y.inverse_transform(validation_dataset.y_values_array)
            y_test_hat_ = scaler_y.inverse_transform(reg_output)
            reg_mae = MAE(y_test_, y_test_hat_)[0]

            print ('abc MAPE: {}\nang_MAPE: {}\ncoords_MAE: {}\nEf_MAE: {}'.format(
                                        abc_mape, ang_mape, sites_mae, reg_mae))


if __name__ == '__main__':
    path_prefix = '/sandbox/vkolluru/Gen_models_for_FANTASTX/CdTe_test_case/Fantastx_GA_rand30/dataset_from_calcs'
    train_PC_array_path = path_prefix + '/train_set_arrays/PC_array.npy'
    train_Y_array_path = path_prefix + '/train_set_arrays/Y_array.npy'
    train_embds_path = path_prefix + '/train_set_arrays/matgl_megnet_embeds.npy'

    test_PC_array_path = path_prefix + '/test_set_arrays/PC_array.npy'
    test_Y_array_path = path_prefix + '/test_set_arrays/Y_array.npy'
    test_embds_path = path_prefix + '/test_set_arrays/matgl_megnet_embeds.npy'
    test_Nsites_path = path_prefix + '/test_set_arrays/Nsites.npy'

    fit(
        gpus = 2,
        max_epochs = 500,
        batch_size = 64,
        print_metrics = True,
        checkpath = None,
        log_every_n_steps = 25,
        enable_model_summary = True,

        train_PC_array_path = train_PC_array_path,
        train_Y_array_path = train_Y_array_path, 
        train_embds_path = train_embds_path,
        test_PC_array_path = test_PC_array_path,
        test_Y_array_path = test_Y_array_path,
        test_embds_path = test_embds_path, 
        test_Nsites_path = test_Nsites_path,
        max_sites = 20,
        zero_pad_rows = 3,

        # NC-VAE params
        input_dim = 148,
        channel_dim = 3,
        regression_dim = 1,
        graph_embds_dim = 16,
        latent_dim = 32,
        max_filters = 128,
        num_filters = None,
        filter_size = [5, 3, 3],
        strides = [2, 2, 1],
        coeffs = (1, 2, 15),
    )
