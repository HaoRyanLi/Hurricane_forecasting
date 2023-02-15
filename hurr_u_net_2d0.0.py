import os
import datetime
import functools
from typing import Tuple, Any, Dict, Sequence
from collections import defaultdict
from tqdm.auto import tqdm
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
print(jax.devices())
import wandb

## Flax (NN in JAX)
import flax
# import tensorflow as tf

from flax import linen as nn
from flax.training import train_state, checkpoints

## Optax (Optimizers in JAX)
import optax

from jax.config import config
# config.update("jax_enable_x64", True)

import flax.linen as nn
import scipy.io
import jax 
import numpy as np

import time, math
import jax
from jax.nn.initializers import normal, zeros
from jax import value_and_grad, vmap, random, jit, lax
import jax.numpy as jnp
from jax.example_libraries import stax, optimizers
import U_net_hurricane as U_net
import h5py
import utilities
from numpy.lib.stride_tricks import sliding_window_view

path = '/work/09012/haoli1/ls6/hurricane/hurricane_data/high_reso/'

use_trunc_data = False
use_fori = True
#! Step : 0 - Generate_data_initilizers
# ? Training inputs
dt = 15*60
batch_size_test = 10
dim_setting = '2d'

# ? Step 0.3 - Spectral method for 2D Navier-Stoke equation initialize parameters
# initialize physic parameters
rad = np.pi/180
Omega = 7.2921e-5

DATASET_PATH = '/work/09012/haoli1/ls6/hurricane/hurricane_data/high_reso/'
CHECKPOINT_PATH = '/work/09012/haoli1/ls6/hurricane/saved_models/'


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

# all_data = np.random.normal(size=[313,Nx_int,Ny_int,222])
Nt, Ny, Nx = all_data.shape[:3]

if use_trunc_data:
    Nx_start = (Nx-Nx_int)//2
    Ny_start = (Ny-Ny_int)//2
    all_data = all_data[:,Ny_start:Ny_start+Ny_int, Nx_start:Nx_start+Nx_int]

num_train = 303
num_test = Nt - num_train
Train_data = all_data[:Nz_int,:num_train]

# Train_times = all_time[:num_train]

# Train_data = utilities.split_train_data(Train_data, split_num)
print("Train data shape: ", Train_data.shape)

print('=' * 20 + ' >>')
print('Loading test data ...')
Test_data = all_data[:Nz_int, num_train:]
# Test_times = all_time[num_train:]
print(Test_data.shape)

x = datetime.datetime.now()
CHECKPOINT_PATH = '/work/09012/haoli1/ls6/hurricane/saved_models/'+str(x)[:10]+'/'

class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include batch statistics
    batch_stats: Any
    train_hparams: Any

class TrainerModule:
    def __init__(self, model_name : str, model_class : nn.Module, model_hparams : dict,
                 optimizer_name : str, lr_scheduler_name : str, optimizer_hparams : dict, exmp_inputs : np.array,
                 train_hparams : dict, with_train_data=True, upload_run=False, seed=42):
        """
        Module for summarizing all training U-net for learning dynamics.

        Inputs:
            model_name - String of the class name, used for logging and saving
            model_class - Class implementing the neural network
            model_hparams - Hyperparameters of the model, used as input to model constructor
            optimizer_name - String of the optimizer name, supporting ['sgd', 'adam', 'adamw']
            optimizer_hparams - Hyperparameters of the optimizer, including learning rate as 'lr'
            exmp_inputs - Example imgs, used as input to initialize the model
            ...
            seed - Seed to use in the model initialization
        """
        super().__init__()
        self.model_name = model_name
        self.model_class = model_class
        self.model_hparams = model_hparams
        self.optimizer_name = optimizer_name
        self.lr_scheduler_name = lr_scheduler_name
        self.optimizer_hparams = optimizer_hparams
        self.train_hparams = train_hparams
        self.with_train_data = with_train_data
        self.upload_run = upload_run
        self.seed = seed
        self.main_rng = jax.random.PRNGKey(self.seed)
        # Create empty model. Note: no parameters yet
        self.model = self.model_class(**self.model_hparams)
        self.log_dir = CHECKPOINT_PATH
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model(exmp_inputs)
        self.upload_wandb()

    def upload_wandb(self):
        # Uploading to wandb
        self.run_name = ('C128_D'+str(num_train)+'_MCa_'+str("%1.0e"%self.train_hparams['mc_u'])+'_Noise_' 
        + str(self.train_hparams['noise_level'])+'_'+self.model_hparams['act_fn_name']+'_N_seq_'+str(self.train_hparams['n_seq'])+'_bs_'
        + str(self.train_hparams['batch_size'])+self.optimizer_name+'_'+self.lr_scheduler_name+str("%1.0e"%self.optimizer_hparams['lr']))
        if self.upload_run:
            wandb.init(project="hurricane-U-net-2d", name=self.run_name)
            wandb.config.model_name = self.model_name
            wandb.config.model_hparams = self.model_hparams
            wandb.config.optimizer_name = self.optimizer_name
            wandb.config.optimizer_lr = self.optimizer_hparams['lr']
            wandb.config.batch_size = self.train_hparams['batch_size']
            wandb.config.n_seq = self.train_hparams['n_seq']
            wandb.config.noise_level = str(self.train_hparams['noise_level'])
            wandb.config.mc_u = self.train_hparams['mc_u']
            wandb.config.dt = self.train_hparams['dt']

    def create_functions(self):
        def add_noise(noise_rng, data):
            print(f"Adding noise to data with noise level {self.train_hparams['noise_level']}")
            noise = jax.random.normal(noise_rng, data.shape)
            data_noise = data + self.train_hparams['noise_level']*noise
            # print(jnp.max(data_noise - data))
            return data_noise

        def rolling_window(a: jnp.ndarray, window: int):
            idx = jnp.arange(len(a) - window + 1)[:, None] + jnp.arange(window)[None, :]
            return a[idx]

        # Transform data to disired shape: (n_train_samples, Nt, Nx) -> (n_train_samples, Nt-n_seq+1, n_seq, Nx)
        def transform_batch_data(batch_data):
            samples = rolling_window(batch_data, self.train_hparams['n_seq']+2)
            return samples

        def squential_loss(i, args):
            loss_ml, loss_mc, u_ml, batch_data, params, batch_stats = args
            outs = self.model.apply({'params': params, 'batch_stats': batch_stats}, 
                                    u_ml, self.train_hparams['dt'], train=True, mutable=['batch_stats'])
            u_ml_out, new_model_state = outs
            batch_stats = new_model_state['batch_stats']

            # primes_init = utilities.gen_primes_init_from_data(u_ml)
            # levelset_init = jnp.tile(Levelset_init, (batch_data.shape[0],1,1,1))
            
            # u_mc0 = utilities.gen_data_from_primes_array(pred_array)
            # u_mc = u_mc0
            # loss_mc += jnp.sum(jnp.mean(jnp.mean((u_mc-u_ml_next)**2, axis=-1), axis=(-1,-2)))/self.train_hparams['batch_size']
            loss_mc +=0 
            # w_mc0 = U_net.single_step_spectral(u_ml)
                # w_mc1 = U_net.single_step_spectral(u_ml_next) - u_ml_next[...,0] + u_ml[...,0]
                # w_mc = 0.5*(w_mc0 + w_mc1)
                # print(f"the w_mc shape {w_mc.shape}")
                # loss_mc += jnp.sum(jnp.mean((w_mc-u_ml_next[...,0])**2, axis=(1,2)))*self.train_hparams['scaling']/self.train_hparams['batch_size']
            # The machine learning term loss
                ## calute the mean equared err for one sample, mean for w,u,v and all subsequences.
                # print(f"the u_ml_next shape {u_ml_next.shape}")
            loss_ml += jnp.mean((u_ml_out[:,1:-1,1:-1]-batch_data[:,i,1:-1,1:-1,:-2])**2)
            u_ml_next = batch_data[:,i].at[:,1:-1,1:-1,:-2].set(u_ml_out[:,1:-1,1:-1])
                # loss_ml = 0
                # print(f"the div shape {U_net.div_free_loss(u_ml_next, axis=(1,2)).shape}")
                # loss_mc_rho += jnp.sum(U_net.div_free_loss(u_ml_next, axis=(1,2)))*self.train_hparams['scaling']/self.train_hparams['batch_size']
            return loss_ml, loss_mc, u_ml_next, batch_data, params, batch_stats
        
        def calculate_loss(params, batch_stats, batch_data, main_rng, train=True):
            # data = utilities.gen_all_data(data)
            print(f"the shape of batch_data: {batch_data.shape}")
            batch_data = rolling_window(batch_data, self.train_hparams['n_seq'])
            print("the shape of batch_data after transformation", batch_data.shape)
            loss_ml = 0
            loss_mc = 0
            
            u_ml = batch_data[:,0]
            
            if self.train_hparams['noise_level'] > 0:
                noise_rng, main_rng = jax.random.split(main_rng)
                u_ml = add_noise(noise_rng, batch_data[:,0])
            # calculate the loss for the next n_seq+1 steps
            loss_ml, loss_mc, u_ml, _, _, batch_stats = lax.fori_loop(1, self.train_hparams['n_seq'], squential_loss,
                                                                    (loss_ml, loss_mc, u_ml, batch_data, params, batch_stats))
            loss = loss_ml + self.train_hparams['mc_u']*loss_mc
            return loss, (loss_ml, loss_mc, batch_stats, main_rng)
        
        @jit
        def train_step(i, args):
            level = i // self.num_steps_per_level
            ind = i % self.num_steps_per_level
            loss, loss_ml, loss_mc, state, main_rng = args
            batch = lax.dynamic_slice_in_dim(self.train_data[level], ind*self.train_hparams['batch_size'], self.train_hparams['batch_size']+self.train_hparams['n_seq']-1)
            loss_fn = lambda params, batch: calculate_loss(params, state.batch_stats, batch, main_rng, train=True)
            rets, gradients = value_and_grad(loss_fn, has_aux=True)(state.params, batch)
            batch_loss, batch_loss_ml, batch_loss_mc, batch_stats, main_rng = rets[0], *rets[1]
            state = state.apply_gradients(grads=gradients, batch_stats=batch_stats)

            loss += batch_loss
            loss_ml += batch_loss_ml
            loss_mc += batch_loss_mc
            return loss, loss_ml, loss_mc, state, main_rng

        # Training function
        def train_epoch(state, main_rng):
            loss, loss_ml, loss_mc = 0, 0, 0
            loss, loss_ml, loss_mc, state, main_rng = lax.fori_loop(0, self.num_steps_per_epoch, train_step, (loss, loss_ml, loss_mc, state, main_rng))
            return loss, loss_ml, loss_mc, state, main_rng

        @jit
        def forward_map(i, args):
            u, state, test_data = args
            u_ml_out = self.model.apply({'params': state.params, 'batch_stats': state.batch_stats}, u, self.train_hparams['dt'], train=False)
            u = test_data[:,i].at[:,1:-1,1:-1,:-2].set(u_ml_out[:,1:-1,1:-1,:])
            return u, state, test_data

        def neural_solver(state, test_data, Nt_test):
            # the shape of test_data: [T, B, H, W, C]	
            u = test_data[:, 0]
            u, _, _ = lax.fori_loop(1, Nt_test, forward_map, (u, state, test_data))
            return u

        def eval_model(state, test_data, n_start=0, n_end=100):
            print(f"The shape of test data: {test_data.shape}")
            u_pred = neural_solver(state, test_data, test_data.shape[1])
            u_true = test_data[:,-1]
            print(f"The shape of u_pred in eval mode: {u_pred.shape}")
            return jnp.mean((u_pred[...,:-2] - u_true[...,:-2])**2)/jnp.mean(u_true[...,:-2]**2)
        
        self.neural_solver = neural_solver
        self.eval_model = jax.jit(eval_model, static_argnames=['n_start', 'n_end'])
        self.train_epoch = train_epoch

    def init_model(self, exmp_inputs):
        # Initialize model
        if self.with_train_data:
            self.train_data = Train_data
        init_rng, self.main_rng = jax.random.split(self.main_rng)
        variables = self.model.init(init_rng, exmp_inputs, self.train_hparams['dt'], train=False)
        self.init_params, self.init_batch_stats = variables['params'], variables['batch_stats']
        self.state = None
    
    def init_optimizer(self, num_epochs):
        # Initialize learning rate schedule and optimizer
        if self.optimizer_name.lower() == 'adam':
            opt_class = optax.adam
        elif self.optimizer_name.lower() == 'adamw':
            opt_class = optax.adamw
        elif self.optimizer_name.lower() == 'sgd':
            opt_class = optax.sgd
        else:
            assert False, f'Unknown optimizer "{opt_class}"'
        # We decrease the learning rate by a factor of 0.1 after 60% and 85% of the training
        init_value=self.optimizer_hparams.pop('lr')
        total_steps = num_epochs*(self.num_steps_per_epoch + 1)
        if self.lr_scheduler_name == 'constant':
            lr_schedule = optax.piecewise_constant_schedule(init_value=init_value,boundaries_and_scales={int(total_steps*0.6): 0.1, int(total_steps*0.85): 0.1})
        elif self.lr_scheduler_name == 'cosine':
            lr_schedule = optax.warmup_cosine_decay_schedule(init_value=init_value, peak_value=10*init_value,warmup_steps=int(total_steps*0.2),
                                                            decay_steps=total_steps, end_value=init_value*1e-1)
        # Clip gradients at max value, and evt. apply weight decay
        transf = [optax.clip(1.0)]
        if opt_class == optax.sgd and 'weight_decay' in self.optimizer_hparams:  # wd is integrated in adamw
            transf.append(optax.add_decayed_weights(self.optimizer_hparams.pop('weight_decay')))
        optimizer = optax.chain(*transf, opt_class(lr_schedule)) if self.optimizer_name.lower() == 'adam' else optax.chain(*transf, opt_class(lr_schedule, **self.optimizer_hparams))
        # Initialize training state
        self.state = TrainState.create(apply_fn=self.model.apply,
                                       params=self.init_params if self.state is None else self.state.params,
                                       batch_stats=self.init_batch_stats if self.state is None else self.state.batch_stats,
                                       train_hparams=self.train_hparams,
                                       tx=optimizer)

    def train_model(self, test_data, num_epochs=1000):
        self.num_steps_per_level = (self.train_data.shape[1]-self.train_hparams['n_seq']+1)//self.train_hparams['batch_size']
        print(f"The num of num_steps_per_level {self.num_steps_per_level}")
        self.num_steps_per_epoch = Nz_int*self.num_steps_per_level
        print(f"The num of num_steps_per_epoch {self.num_steps_per_epoch}")
        self.init_optimizer(num_epochs)
        err_test_min = 1e4
        epoch_min = -1
        for epoch_idx in tqdm(range(1, num_epochs+1)):
            t1 = time.time()
            loss, loss_ml, loss_mc, self.state, self.main_rng = self.train_epoch(self.state, self.main_rng)
            err_test = self.eval_model(self.state, test_data)
            t2 = time.time()
            if err_test_min >= err_test:
                err_test_min = err_test
                epoch_min = epoch_idx
                self.save_model(step=epoch_idx)
            if epoch_idx % 100 == 0:  # Print MSE every 100 epochs
                print("n_seq {:d}, batch {:d}, mc_u {:.2e}, time {:.2e}s, loss {:.2e}, ml_loss {:.2e}, mc_loss {:.2e}, TE {:.2e}, TE_min {:.2e}, EPmin {:d}, EP {}".format(
                          self.train_hparams['n_seq'], self.train_hparams['batch_size'], self.train_hparams['mc_u'],
                        t2 - t1, loss, loss_ml, loss_mc, err_test, err_test_min, epoch_min, epoch_idx))
            if self.upload_run:
                wandb.log({"Total loss": float(loss), "ML loss": loss_ml, "MC loss": loss_mc, "Test Error": err_test, 'TEST MIN': err_test_min})

    def save_model(self, step=0):
        self.log_dir = os.path.join(CHECKPOINT_PATH, self.run_name)
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir, keep=5, target={'params': self.state.params, 'batch_stats': self.state.batch_stats,
                                                                   'train_hparams': self.train_hparams}, step=step, overwrite=True)
        
    def load_model(self, pretrained=False):
        if not pretrained:
        # Load a specific model. Usually for test and plot.
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=None)
        else:
            # Load a model the same as current setting. Usually for continu training.
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(CHECKPOINT_PATH, f'{self.model_name}.ckpt'), target=None)
        self.state = TrainState.create(apply_fn=self.model.apply, params=state_dict['params'], batch_stats=flax.core.frozen_dict.freeze(state_dict['batch_stats']), 
                                       train_hparams=state_dict['train_hparams'], tx=self.state.tx if self.state else optax.sgd(0.1))
        self.train_hparams = self.state.train_hparams
    
    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this autoencoder
        return os.path.isfile(os.path.join(CHECKPOINT_PATH, f'{self.model_name}.ckpt'))

trainer = TrainerModule(model_name="UNet", model_class=U_net.UNet,
                        model_hparams={"act_fn": nn.relu, "act_fn_name": 'relu',  "padding": "REPLICATE", "block_size": (8, 16, 32, 64, 128, 128, 128), 
                        "out_features": Train_data.shape[-1]-2, "model_type": "U_net_modified", "f_cori":f_cori, "Nc_uv":Nc_dim}, 
                        optimizer_name="adam", lr_scheduler_name="cosine", optimizer_hparams={"lr": 1e-4,"weight_decay": 1e-4},
                        exmp_inputs=jax.device_put(Train_data[0,:1]),
                        train_hparams={'batch_size':5, 'n_seq':5, 'mc_u':1, 'dt':dt, 'noise_level':0.0, 'scaling': 1},
                        upload_run=True)
# print("loading pre-trained model")
# trainer.load_model()
# print("Sucessfully loaded pre-trained modepythol")

# print(trainer.eval_model(trainer.state, Test_data))

num_epochs=10
print(f"training new model, the num of training epochs is {num_epochs}")
trainer.train_model(Test_data, num_epochs=num_epochs)
