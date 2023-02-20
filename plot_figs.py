import os
import datetime
import functools
from typing import Tuple, Any, Dict, Sequence
from collections import defaultdict
from tqdm.auto import tqdm
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

from jax.config import config
import flax.linen as nn
import scipy.io
import jax 
import numpy as np
import jax.numpy as jnp
import h5py
import utilities
from numpy.lib.stride_tricks import sliding_window_view
import U_net_hurricane as U_net

import matplotlib.pyplot as plt
import matplotlib

import tkinter
matplotlib.use('TkAgg')


use_trunc_data = False
use_fori = False
dim_setting = '2d'


dt_scaling = 15*60
dt = 15*60/dt_scaling
norm_paras = {'U':(-10, 10), 'V':(-10, 10), 'D':(0.9, 1.2), 'P':(80000, 105000), 'x':(-98.15747, -87.685425), 'y':(23.304024, 30.74073)}

batch_size_test = 5

Nx_int = 512
Ny_int = 384
Nz_int = 10

scal_factor = 500

rad = np.pi/180
Omega = 7.2921e-5

if dim_setting == '3d':
    Num_level = 1
    Nc_dim = Nz_int
    from Train_mod_3d import TrainerModule
elif dim_setting == '2d':
    Num_level = Nz_int
    Nc_dim = 1
    from Train_mod_2d import TrainerModule

DATASET_PATH = '/work/09012/haoli1/ls6/hurricane/hurricane_data/high_reso/'
train_file_name = DATASET_PATH+'Nt_313_10x512x384_uvdp_int_2023-02-09.h5'

CHECKPOINT_PATH = '/work/09012/haoli1/ls6/hurricane/saved_models/3d_model/2023-02-19/2D_F_0_dt_1e+00_D300_Scal500_Noise_0.0_relu_N_seq_5_bs_10adam_cosine1e-04'
    
hf_data = h5py.File(train_file_name, 'r')
all_data, xx_norm, yy_norm = utilities.norm_data(hf_data, norm_paras, Nz_int, dim_set=dim_setting)
all_data = jax.device_put(all_data)
xx, yy = utilities.recover_norm_data(norm_paras, ['x', 'y'], [xx_norm, yy_norm])

if use_fori:
    f_cori = 2*Omega*jnp.sin(yy*rad)
else:
    f_cori = np.zeros_like(yy)

Nt, Ny, Nx = all_data.shape[:3]

if use_trunc_data:
    Nx_start = (Nx-Nx_int)//2
    Ny_start = (Ny-Ny_int)//2
    all_data = all_data[:,Ny_start:Ny_start+Ny_int, Nx_start:Nx_start+Nx_int]

num_train = 253
num_test = Nt - num_train

if dim_setting == '3d':
    Train_data = all_data[:num_train]
    Test_data = all_data[num_train:]
elif dim_setting == '2d':
    Train_data = all_data[:, :num_train]
    Test_data = all_data[:, num_train:]
print("Train data shape: ", Train_data.shape)
print("Test data shape: ", Test_data.shape)
del all_data

if dim_setting == '3d':
    trainer = TrainerModule(project="hurricane-U-net", model_name="UNet", model_class=U_net.UNet,
                            model_hparams={"act_fn": nn.relu, "act_fn_name": 'relu',  "padding": "REPLICATE", "block_size": (16, 32, 64, 128, 128, 128, 128), 
                            "out_features": Train_data.shape[-1]-2, "model_type": "U_net_modified", "f_cori":f_cori, "Nc_uv":Nc_dim}, 
                            optimizer_name="adam", lr_scheduler_name="cosine", optimizer_hparams={"lr": 1e-4,"weight_decay": 1e-4},
                            exmp_inputs=jax.device_put(Train_data[:1]),
                            train_hparams={'batch_size':5, 'n_seq':5, 'mc_u':1, 'dt':dt, 'noise_level':0.0}, num_train=Train_data.shape[1], 
                            check_pt=CHECKPOINT_PATH, norm_paras=norm_paras, batch_size_test=batch_size_test, use_fori=use_fori, num_level=Num_level,
                            scal_fact=scal_factor, upload_run=False)

elif dim_setting == '2d':
    trainer = TrainerModule(project="hurricane-U-net-2d", model_name="UNet", model_class=U_net.UNet,
                        model_hparams={"act_fn": nn.relu, "act_fn_name": 'relu',  "padding": "REPLICATE", "block_size": (8, 16, 32, 64, 128, 128, 128), 
                        "out_features": Train_data.shape[-1]-2, "model_type": "U_net_modified", "f_cori":f_cori, "Nc_uv":Nc_dim}, 
                        optimizer_name="adam", lr_scheduler_name="cosine", optimizer_hparams={"lr": 1e-4,"weight_decay": 1e-4},
                        exmp_inputs=jax.device_put(Train_data[0,:1]),
                        train_hparams={'batch_size':10, 'n_seq':5, 'mc_u':1, 'dt':dt, 'noise_level':0.0}, num_train=Train_data.shape[1], 
                        check_pt=CHECKPOINT_PATH, norm_paras=norm_paras, use_fori=use_fori, num_level=Nz_int, scal_fact=scal_factor,
                        upload_run=True)

print("loading pre-trained model")
trainer.load_model()
print("Sucessfully loaded pre-trained modepythol")
print(trainer.eval_model(trainer.state, Test_data))

test_data = Test_data
batch_size_test = 1
Nt_test = 20
num_test = (Nt_test+1) * batch_size_test
if dim_setting == '3d':
    N, H, W, C = test_data.shape
    test_data = test_data[:num_test].reshape((batch_size_test, Nt_test+1, H, W, C))
elif dim_setting == '2d':
    test_data = test_data[:batch_size_test,:num_test]

print(f"The shape of test data: {test_data.shape}")
nn_sol_all = trainer.neural_solver(trainer.state, test_data, Nt_test+1)[0]


## to slice the true solution
true_sol_all = test_data[0,Nt_test]

print(f"dt: {trainer.train_hparams['dt']}")

norm_U_nn = nn_sol_all[...,:Nc_dim]
norm_V_nn = nn_sol_all[...,Nc_dim:2*Nc_dim]
norm_D_nn = nn_sol_all[...,2*Nc_dim:3*Nc_dim]
norm_P_nn = nn_sol_all[...,3*Nc_dim:4*Nc_dim]
norm_data_nn = [norm_U_nn, norm_V_nn, norm_D_nn, norm_P_nn]

norm_U_true = true_sol_all[...,:Nc_dim]
norm_V_true = true_sol_all[...,Nc_dim:2*Nc_dim]
norm_D_true = true_sol_all[...,2*Nc_dim:3*Nc_dim]
norm_P_true = true_sol_all[...,3*Nc_dim:4*Nc_dim]
norm_data_true = [norm_U_true, norm_V_true, norm_D_true, norm_P_true]


utilities.get_max_min(true_sol_all, Nc_dim)

keys = ['U', 'V', 'D', 'P']
for i in range(4):
    key = keys[i]
    err_term = norm_data_true[i]-norm_data_nn[i]
    err = jnp.mean(err_term**2)
    rel_err = err/jnp.mean(norm_data_true[i]**2)
    print(f"The err and relative error of normed {key} is {err} and {rel_err}, the max err value is {np.max(err_term)}, the min err value is {np.min(err_term)}")

U_nn, V_nn, D_nn, P_nn = utilities.recover_norm_data(norm_paras, keys, norm_data_nn)
data_nn = [U_nn, V_nn, D_nn, P_nn]
U_true, V_true, D_true, P_true = utilities.recover_norm_data(norm_paras, keys, norm_data_true)
data_true = [U_true, V_true, D_true, P_true]

for i in range(4):
    key = keys[i]
    err_term = data_true[i]-data_nn[i]
    rel_err = jnp.mean(err_term**2)/jnp.mean(data_true[i]**2)
    print(f"The relative error of real {key} is {rel_err}, the max err value is {np.max(err_term)}, the min err value is {np.min(err_term)}")

print(jnp.mean((nn_sol_all-true_sol_all)**2)/jnp.mean(true_sol_all**2))


level = 0
var2ind = {'U':0+level, 'V':Nc_dim+level, 'D':2*Nc_dim+level, 'P':3*Nc_dim+level}
imag_path = f'/work/09012/haoli1/ls6/hurricane/figs/{dim_setting}_model_figs/'

x_start = np.max(xx[:,0])
x_end = np.min(xx[:,-1])
y_start = np.max(yy[0])
y_end = np.min(yy[-1])
plot_show = True

for i in range(4):
    var = keys[i]
    init = test_data[0,0,:,:,var2ind[var]]
    init = utilities.recover_norm_data(norm_paras, [var], [init])[0]
    Z0min = init.min()
    Z0max = abs(init).max()

    nn_sol = data_nn[i][...,level]
    true_sol = data_true[i][...,level]
    utilities.plot_fig(init, f"True {var} at T={0}hrs", Z0min, Z0max, x_start, x_end, y_start, y_end, 
                        imag_path+f'{var}_init_{dim_setting}.eps', plot_show=plot_show)
    utilities.plot_fig(nn_sol, f"Predicted {var} at T={Nt_test/4}hrs", Z0min, Z0max, x_start, x_end, y_start, y_end, 
                        imag_path+f'{var}_nn_{str(int(Nt_test/4))}hr_{dim_setting}.eps', plot_show=plot_show)
    utilities.plot_fig(true_sol, f"True {var} at T={Nt_test/4}hrs", Z0min, Z0max, x_start, x_end, y_start, y_end, 
                        imag_path+f'{var}_true_{str(int(Nt_test/4))}hr_{dim_setting}.eps', plot_show=plot_show)

    err = nn_sol-true_sol
    err_min=err.min()
    err_max=abs(err).max()
    utilities.plot_fig(err, f"{var} error at T={Nt_test/4}hrs", err_min, err_max, x_start, x_end, y_start, y_end, 
                        imag_path+f'err_{var}_{str(int(Nt_test/4))}hr_{dim_setting}.eps', plot_show=plot_show)

