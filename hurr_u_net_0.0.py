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

from Train_mod_3d import TrainerModule
import U_net_hurricane as U_net


use_trunc_data = False
use_fori = False
dim_setting = '3d'
dt_scaling = 15*60
dt = 15*60/dt_scaling
norm_paras = {'U':(-10, 10), 'V':(-10, 10), 'D':(0.9, 1.2), 'P':(80000, 105000), 'x':(-98.15747, -87.685425), 'y':(23.304024, 30.74073)}

batch_size_test =5

Nx_int = 512
Ny_int = 384
Nz_int = 10

scal_factor = 100

rad = np.pi/180
Omega = 7.2921e-5
DATASET_PATH = '/work/09012/haoli1/ls6/hurricane/hurricane_data/high_reso/'
x = datetime.datetime.now()
CHECKPOINT_PATH = '/work/09012/haoli1/ls6/hurricane/saved_models/3d_model/'+str(x)[:10]+'/'

if dim_setting == '3d':
    Num_level = 1
    Nc_dim = Nz_int
elif dim_setting == '2d':
    Num_level = Nz_int
    Nc_dim = 1

scal_vec = np.concatenate([scal_factor*np.ones(Nc_dim), scal_factor*np.ones(Nc_dim), np.ones(Nc_dim), np.ones(Nc_dim)])
scal_vec = scal_vec/np.sqrt(np.sum(scal_vec**2))
print(f"The norm of scal_vec is {np.dot(scal_vec, scal_vec)}")

train_file_name = DATASET_PATH+'Nt_313_10x512x384_uvdp_int_2023-02-09.h5'

print('=' * 20 + ' >>')
print('Loading train data ...')
    
hf_data = h5py.File(train_file_name, 'r')
all_data, xx_norm, yy_norm = utilities.norm_data(hf_data, norm_paras, Nz_int, dim_set=dim_setting)
all_data = jax.device_put(all_data)
yy = utilities.recover_norm_data(norm_paras, ['y'], [yy_norm])[0]

if use_fori:
    f_cori = 2*Omega*jnp.sin(yy*rad)
else:
    f_cori = np.zeros_like(yy)

# all_data = np.random.normal(size=[313,Nx_int,Ny_int,222])
Nt, Ny, Nx = all_data.shape[:3]

if use_trunc_data:
    Nx_start = (Nx-Nx_int)//2
    Ny_start = (Ny-Ny_int)//2
    all_data = all_data[:,Ny_start:Ny_start+Ny_int, Nx_start:Nx_start+Nx_int]

num_train = 253
num_test = Nt - num_train
Train_data = all_data[:num_train]

# Train_times = all_time[:num_train]

# Train_data = utilities.split_train_data(Train_data, split_num)
print("Train data shape: ", Train_data.shape)

print('=' * 20 + ' >>')
print('Loading test data ...')
Test_data = all_data[num_train:]
del all_data

# Test_times = all_time[num_train:]
print(Test_data.shape)



trainer = TrainerModule(project="hurricane-U-net", model_name="UNet", model_class=U_net.UNet,
                        model_hparams={"act_fn": nn.relu, "act_fn_name": 'relu',  "padding": "REPLICATE", "block_size": (16, 32, 64, 128, 128, 128, 128), 
                        "out_features": Train_data.shape[-1]-2, "model_type": "U_net_modified", "f_cori":f_cori, "Nc_uv":Nc_dim}, 
                        optimizer_name="adam", lr_scheduler_name="cosine", optimizer_hparams={"lr": 1e-4,"weight_decay": 1e-4},
                        exmp_inputs=jax.device_put(Train_data[:1]),
                        train_hparams={'batch_size':5, 'n_seq':5, 'mc_u':1, 'dt':dt, 'noise_level':0.0}, num_train=Train_data.shape[1], 
                        check_pt=CHECKPOINT_PATH, norm_paras=norm_paras, batch_size_test=batch_size_test, use_fori=use_fori, num_level=Num_level,
                        scal_fact=scal_vec, upload_run=True)


# print("loading pre-trained model")
# trainer.load_model()
# print("Sucessfully loaded pre-trained modepythol")
# print(trainer.eval_model(trainer.state, Test_data))

num_epochs=5
print(f"training new model, the num of training epochs is {num_epochs}")
trainer.train_model(Train_data, Test_data, num_epochs=num_epochs)
import os, psutil; print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)