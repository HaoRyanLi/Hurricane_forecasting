import os
from typing import Tuple, Any, Dict, Sequence
from collections import defaultdict
from tqdm.auto import tqdm
from jax.lib import xla_bridge
import wandb
import flax
from flax.training import train_state, checkpoints
import flax.linen as nn
import scipy.io
import jax 
import numpy as np
from jax.nn.initializers import normal, zeros
from jax import value_and_grad, vmap, random, jit, lax
import jax.numpy as jnp
from jax.example_libraries import stax, optimizers
import U_net_hurricane as U_net
import optax
import time, math
import utilities





class TrainState(train_state.TrainState):
    # A simple extension of TrainState to also include batch statistics
    batch_stats: Any
    train_hparams: Any

class TrainerModule:
    def __init__(self, project: str, model_name: str, model_class: nn.Module, model_hparams: dict,
                 optimizer_name: str, lr_scheduler_name: str, optimizer_hparams: dict, exmp_inputs: np.array,
                 train_hparams: dict, num_train: int, check_pt: str, norm_paras: dict,
                 batch_size_test=5,
                 use_fori=False, num_level=1, with_train_data=True, upload_run=False, seed=42):
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
        self.project = project
        self.model_name = model_name
        self.model_class = model_class
        self.model_hparams = model_hparams
        self.optimizer_name = optimizer_name
        self.lr_scheduler_name = lr_scheduler_name
        self.optimizer_hparams = optimizer_hparams
        self.train_hparams = train_hparams
        self.num_train = num_train
        self.batch_size_test = batch_size_test
        self.use_fori = 1 if use_fori else 0
        self.num_level = num_level
        self.with_train_data = with_train_data
        self.upload_run = upload_run
        self.seed = seed
        self.main_rng = jax.random.PRNGKey(self.seed)
        # Create empty model. Note: no parameters yet
        self.model = self.model_class(**self.model_hparams)
        self.log_dir = check_pt
        self.check_pt = check_pt
        self.norm_paras = norm_paras
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model(exmp_inputs)
        self.upload_wandb()

    def upload_wandb(self):
        # Uploading to wandb
        self.run_name = ('F_'+str(self.use_fori)+'_dt_'+str("%1.0e"%self.train_hparams['dt'])+'_D'+str(self.num_train)+'_MCa_'+str("%1.0e"%self.train_hparams['mc_u'])+'_Noise_' 
        + str(self.train_hparams['noise_level'])+'_'+self.model_hparams['act_fn_name']+'_N_seq_'+str(self.train_hparams['n_seq'])+'_bs_'
        + str(self.train_hparams['batch_size'])+self.optimizer_name+'_'+self.lr_scheduler_name+str("%1.0e"%self.optimizer_hparams['lr']))
        if self.upload_run:
            wandb.init(project=self.project, name=self.run_name)
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
            outs = self.model.apply({'params': params, 'batch_stats': batch_stats}, u_ml, self.train_hparams['dt'], train=True, mutable=['batch_stats'])
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
            print(f"The shape of u_ml_next: {u_ml_next.shape}")
                # loss_ml = 0
                # print(f"the div shape {U_net.div_free_loss(u_ml_next, axis=(1,2)).shape}")
                # loss_mc_rho += jnp.sum(U_net.div_free_loss(u_ml_next, axis=(1,2)))*self.train_hparams['scaling']/self.train_hparams['batch_size']
            return loss_ml, loss_mc, u_ml_next, batch_data, params, batch_stats
        
        def calculate_loss(params, batch_stats, batch_data, main_rng, train=True):
            # data = utilities.gen_all_data(data)
            print(f"the shape of batch_data: {batch_data.shape}")
            batch_data = rolling_window(batch_data, self.train_hparams['n_seq'])
            print("the shape of batch_data after transformation", batch_data.shape)
            loss_ml = 0.0
            loss_mc = 0.0
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
            loss, loss_ml, loss_mc, state, main_rng = args
            batch = lax.dynamic_slice_in_dim(self.train_data, i*self.train_hparams['batch_size'], self.train_hparams['batch_size']+self.train_hparams['n_seq']-1)
            loss_fn = lambda params, batch: calculate_loss(params, state.batch_stats, batch, main_rng, train=True)
            rets, gradients = value_and_grad(loss_fn, has_aux=True)(state.params, batch)
            batch_loss, batch_loss_ml, batch_loss_mc, batch_stats, main_rng = rets[0], *rets[1]
            state = state.apply_gradients(grads=gradients, batch_stats=batch_stats)

            loss += batch_loss
            loss_ml += batch_loss_ml
            loss_mc += batch_loss_mc
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
            N, H, W, C = test_data.shape
            Nt_test = N//self.batch_size_test
            num_test = Nt_test * self.batch_size_test
            test_data = test_data[:num_test].reshape((self.batch_size_test, Nt_test, H, W, C))
            print(f"The shape of test data: {test_data.shape}")
            u_pred = neural_solver(state, test_data, Nt_test)
            u_true = test_data[:,-1]
            rel_err_u, rel_err_v, rel_rr_d, rel_err_p, rel_err_sum = utilities.get_real_rel_err(self.norm_paras, u_pred, u_true, self.model_hparams['Nc_uv'])
            return rel_err_u, rel_err_v, rel_rr_d, rel_err_p, rel_err_sum
        
        self.neural_solver = neural_solver
        self.eval_model = jax.jit(eval_model, static_argnames=['n_start', 'n_end'])
        self.train_step = train_step

    def init_model(self, exmp_inputs):
        print(f"The shape of init input: {exmp_inputs.shape}")
        init_rng, self.main_rng = jax.random.split(self.main_rng)
        variables = self.model.init(init_rng, exmp_inputs, self.train_hparams['dt'], train=True)
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
            lr_schedule = optax.warmup_cosine_decay_schedule(init_value=init_value, peak_value=5*init_value,warmup_steps=int(total_steps*0.25),
                                                            decay_steps=total_steps, end_value=init_value*1e-1)
        self.lrs = jnp.stack([lr_schedule(i) for i in range(total_steps)])
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
    def train_epoch(self):
        loss, loss_ml, loss_mc = 0, 0, 0
        loss, loss_ml, loss_mc, self.state, self.main_rng = lax.fori_loop(0, self.num_steps_per_epoch, self.train_step, 
                                                                (loss, loss_ml, loss_mc, self.state, self.main_rng))
        loss, loss_ml, loss_mc = loss/self.num_steps_per_epoch, loss_ml/self.num_steps_per_epoch, loss_mc/self.num_steps_per_epoch
        return loss, loss_ml, loss_mc


    def train_model(self, train_data, test_data, num_epochs):
        self.train_data = train_data
        self.num_steps_per_epoch = (train_data.shape[0]-self.train_hparams['n_seq']+1)//self.train_hparams['batch_size']	

        self.init_optimizer(num_epochs)
        err_test_min = 1e10
        epoch_min = -1
        for epoch_idx in tqdm(range(1, num_epochs+1)):
            loss, loss_ml, loss_mc = self.train_epoch()
            rel_err_u, rel_err_v, rel_rr_d, rel_err_p, err_test = self.eval_model(self.state, test_data)
            print(rel_err_u, rel_err_v, rel_rr_d, rel_err_p, err_test)
            if err_test_min >= err_test:
                err_test_min = err_test
                epoch_min = epoch_idx
                self.save_model(step=epoch_idx)
            if epoch_idx % 100 == 0:  # Print MSE every 100 epochs
                print("n_seq {:d}, batch {:d}, mc_u {:.2f}, loss {:.2f}, ml_loss {:.2f}, mc_loss {:.2f}, TE {:.2f}, TE_min {:.2f}, EPmin {:d}, EP {:d}".format(
                        self.train_hparams['n_seq'], self.train_hparams['batch_size'], self.train_hparams['mc_u'], loss, loss_ml, loss_mc, err_test, err_test_min,
                        epoch_min, epoch_idx))
            if self.upload_run:
                wandb.log({"Total Loss": float(loss), "Test Error": err_test, 'TEST MIN': err_test_min,
                            'U Err':float(rel_err_u), 'V Err': rel_err_v, 'D Err': rel_rr_d, 'P Err': rel_err_p})
    def save_model(self, step=0):
        self.log_dir = os.path.join(self.check_pt, self.run_name)
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir, keep=5, target={'params': self.state.params, 'batch_stats': self.state.batch_stats,
                                                                   'train_hparams': self.train_hparams}, step=step, overwrite=True)
        
    def load_model(self, pretrained=False):
        if not pretrained:
        # Load a specific model. Usually for test and plot.
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=None)
        else:
            # Load a model the same as current setting. Usually for continu training.
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(self.check_pt, f'{self.model_name}.ckpt'), target=None)
        self.state = TrainState.create(apply_fn=self.model.apply, params=state_dict['params'], batch_stats=flax.core.frozen_dict.freeze(state_dict['batch_stats']), 
                                       train_hparams=state_dict['train_hparams'], tx=self.state.tx if self.state else optax.sgd(0.1))
        self.train_hparams = self.state.train_hparams
    
    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this autoencoder
        return os.path.isfile(os.path.join(self.check_pt, f'{self.model_name}.ckpt'))