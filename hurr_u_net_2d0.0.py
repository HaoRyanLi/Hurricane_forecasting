import os
import datetime
import functools
from typing import Tuple, Any, Dict, Sequence
from collections import defaultdict
from tqdm.auto import tqdm
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

from jax.config import config
# config.update("jax_enable_x64", True)

import flax.linen as nn
import scipy.io
import jax 
import numpy as np
import jax.numpy as jnp
import h5py
import utilities
from numpy.lib.stride_tricks import sliding_window_view

from Train_mod_2d import TrainerModule
import U_net_hurricane as U_net


use_trunc_data = False
use_fori = True
#! Step : 0 - Generate_data_initilizers
# ? Training inputs
dt_scaling = 100
dt = 15*60/dt_scaling

batch_size_test = 10
dim_setting = '2d'

# ? Step 0.3 - Spectral method for 2D Navier-Stoke equation initialize parameters
# initialize physic parameters
rad = np.pi/180
Omega = 7.2921e-5

DATASET_PATH = '/work/09012/haoli1/ls6/hurricane/hurricane_data/high_reso/'
x = datetime.datetime.now()
CHECKPOINT_PATH = '/work/09012/haoli1/ls6/hurricane/saved_models/'+str(x)[:10]+'/'

Nx_int = 512
Ny_int = 384
Nz_int = 10
if dim_setting == '3d':
    Nc_dim = Nz_int
elif dim_setting == '2d':
    Nc_dim = 1

#! Step 1: Loading data
# ? 1.1 Loading data by pandas
print('=' * 20 + ' >>')
print('Loading train data ...')

train_file_name = DATASET_PATH+'Nt_313_10x512x384_uvdp_int_2023-02-09.h5'
    
hf_data = h5py.File(train_file_name, 'r')

all_data, norm_paras, xx_norm, yy_norm = utilities.norm_data(hf_data, Nz_int, dim_set=dim_setting)
yy = utilities.recover_data(norm_paras, ['y'], [yy_norm])[0]
if use_fori:
    f_cori = 2*Omega*jnp.sin(yy*rad)
else:
    f_cori = np.zeros_like(yy)

f_cori = f_cori*dt_scaling
# all_data = np.random.normal(size=[313,Nx_int,Ny_int,222])
Nt, Ny, Nx = all_data.shape[:3]

if use_trunc_data:
    Nx_start = (Nx-Nx_int)//2
    Ny_start = (Ny-Ny_int)//2
    all_data = all_data[:,Ny_start:Ny_start+Ny_int, Nx_start:Nx_start+Nx_int]

num_train = 300
num_test = Nt - num_train
all_data = jax.device_put(all_data)
Train_data = all_data[:Nz_int, :num_train]

print("Train data shape: ", Train_data.shape)

print('=' * 20 + ' >>')
print('Loading test data ...')
Test_data = all_data[:Nz_int, num_train:]

del all_data
print(Test_data.shape)


trainer = TrainerModule(project="hurricane-U-net-2d", model_name="UNet", model_class=U_net.UNet,
                        model_hparams={"act_fn": nn.relu, "act_fn_name": 'relu',  "padding": "REPLICATE", "block_size": (8, 16, 32, 64, 128, 128, 128), 
                        "out_features": Train_data.shape[-1]-2, "model_type": "U_net_modified", "f_cori":f_cori, "Nc_uv":Nc_dim}, 
                        optimizer_name="adam", lr_scheduler_name="cosine", optimizer_hparams={"lr": 1e-4,"weight_decay": 1e-4},
                        exmp_inputs=jax.device_put(Train_data[0,:1]),
                        train_hparams={'batch_size':10, 'n_seq':5, 'mc_u':1, 'dt':dt, 'noise_level':0.0}, num_train=Train_data.shape[1], 
                        check_pt=CHECKPOINT_PATH, use_fori=use_fori, num_level=Nz_int,
                        upload_run=True)

# print("loading pre-trained model")
# trainer.load_model()
# print("Sucessfully loaded pre-trained modepythol")
# print(trainer.eval_model(trainer.state, Test_data))

num_epochs=3
print(f"training new model, the num of training epochs is {num_epochs}")
trainer.train_model(Train_data, Test_data, num_epochs=num_epochs)
import os, psutil; print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)