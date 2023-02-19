import os
import functools
from typing import Tuple, Any, Dict, Sequence
from collections import defaultdict
from tqdm.auto import tqdm

## Flax (NN in JAX)
import flax
# import tensorflow as tf

from flax import linen as nn

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
import utilities


path = '/work/09012/haoli1/ls6/hurricane/hurricane_data/high_reso/'

dim_setting = 2
if dim_setting == 2:
    pad_width_setting = ((0,0),(1,1),(1,1),(0,0))
elif dim_setting == 3:
    pad_width_setting = ((0,0),(1,1),(1,1),(1,1),(0,0))

# @jax.jit
# def gen_primes_init(x):
#     nh = halo_cell_num
#     (B, nx, ny, C) = x.shape
#     nz = 1
#     index = [0, 1, 2, 4]
#     data = jnp.stack([x[...,0], x[...,1], x[...,2], jnp.zeros_like(x[...,0]), x[...,3]], axis=1)
#     data = jnp.expand_dims(data, axis=-1)
#     nhx = jnp.s_[halo_cell_num:-halo_cell_num]
#     primes = jnp.ones((B, 5,nx + 2*nh if nx > 1 else nx, ny + 2*nh if ny > 1 else ny, nz + 2*nh if nz > 1 else nz))*initializer.eps
#     primes = primes.at[...,nhx,nhx,nhx].set(data)
#     return primes

# @jax.jit
# def gen_conv_data(primes):
#     index_set = [0, 1, 2, 4]
#     conv_data = jnp.stack([primes[:,i] for i in index_set], axis=-1)
#     return jnp.squeeze(conv_data)

# def fill_halo_primes_and_cons(primes, current_time, initializer):
#     '''
#     The shape of primes_init: [B, 5, nx, ny, nz]
#     '''
#     cons = get_conservatives_from_primitives(primes, initializer.material_manager)
#     cons, primes = initializer.boundary_condition.fill_boundary_primes(cons, primes, current_time)
#     return primes
# vmap_fill_halo_primes_and_cons = jax.vmap(fill_halo_primes_and_cons, in_axes=(0, None, None))

class Conv_repl(nn.Module):
    """Convolution layer for with replicate padding.
    Attributes:
        features: Num convolutional features.
    """
    features: int
    kernel_size: tuple = (3, 3)
    pad_width: tuple = pad_width_setting
    mode: str = 'edge'
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        activation = jnp.pad(x, pad_width=self.pad_width, mode=self.mode)
        activation = nn.Conv(features=self.features, kernel_size=self.kernel_size, name='conv1', padding='VALID')(activation)
        return activation

class DeConv3x3(nn.Module):
    """Deconvolution layer for upscaling.
    Attributes:
        features: Num convolutional features.
        padding: Default type of padding: 'SAME'.
        use_batch_norm: Whether to use batchnorm at the end or not.
    """
    features: int
    padding: str = 'SAME'
    use_batch_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        """Applies deconvolution with 3x3 kernel."""
        (B, H, W, C) = x.shape
        x = jax.image.resize(x, shape=(B, H*2, W*2, C), method='bilinear')
        if self.padding == 'REPLICATE':
            x = Conv_repl(features=self.features, kernel_size=(3, 3), name='deconv_repl')(x)
        else:
            x = nn.Conv(features=self.features, kernel_size=(3, 3), name='deconv_'+self.padding, padding=self.padding)(x)
        # if self.use_batch_norm:
        #   x = nn.BatchNorm(use_running_average=not train)(x)
        return x

class ConvLayer2(nn.Module):
    """Two unpadded convolutions & activation.
    Attributes:
        features: Num convolutional features.
        act_fn: activiation function 
        padding: Default type of padding: 'SAME'.
        use_batch_norm: Whether to use batchnorm at the end or not.
    """
    features: int
    act_fn: callable = nn.tanh
    padding: str = 'SAME'
    use_batch_norm: bool = True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        if self.padding == 'REPLICATE':
            activation = Conv_repl(features=self.features, kernel_size=(3, 3), name='conv1_repl')(x)
        else:
            activation = nn.Conv(features=self.features, kernel_size=(3, 3), name='conv1_'+self.padding, padding=self.padding)(x)
        
        if self.use_batch_norm:
            activation = nn.BatchNorm(use_running_average=not train)(activation)
        activation = self.act_fn(activation)

        if self.padding == 'REPLICATE':
            activation = Conv_repl(features=self.features, kernel_size=(3, 3), name='conv2_repl')(activation)
        else:
            activation = nn.Conv(features=self.features, kernel_size=(3, 3), name='conv2_'+self.padding, padding=self.padding)(activation)
        
        if self.use_batch_norm:
            activation = nn.BatchNorm(use_running_average=not train)(activation)
        activation = self.act_fn(activation)
        return activation

class DownsampleBlock(nn.Module):
    """Two unpadded convolutions & downsample 2x.
    Attributes:
        features: Num convolutional features.
        act_fn: activiation function 
        padding: Default type of padding: 'SAME'.
        use_batch_norm: Whether to use batchnorm at the end or not.
    """

    features: int
    act_fn: callable = nn.tanh
    padding: str = 'SAME'
    use_batch_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        residual = x = ConvLayer2(features=self.features, act_fn=self.act_fn, padding=self.padding, use_batch_norm=self.use_batch_norm)(x, train=train)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x, residual

class BottleneckBlock(nn.Module):
    """Two unpadded convolutions, dropout & deconvolution.
    Attributes:
        features: Num convolutional features.
        layers_num: List of Num of FNN layers.
        act_fn: activiation function 
    """
    features: int
    layers_num: list
    act_fn: callable = nn.tanh
    padding: str = 'SAME'
    use_batch_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        x = ConvLayer2(self.features, act_fn=self.act_fn, padding=self.padding, use_batch_norm=self.use_batch_norm)(x, train=train)
        return x

class BottleneckBlock_modified(nn.Module):
    """Two unpadded convolutions, dropout & deconvolution.
    Attributes:
        features: Num convolutional features.
        layers_num: List of Num of FNN layers.
        act_fn: activiation function 
    """
    features: int
    layers_num: list
    act_fn: callable = nn.tanh
    padding: str = 'SAME'
    use_batch_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        (B, H, W, C) = x.shape
        activation = x.reshape((B, H*W*C))
        for i, n in enumerate(self.layers_num):
            activation = nn.Dense(self.layers_num[i])(activation)
            activation = self.act_fn(activation)
        activation = nn.Dense(H*W*C)(activation)
        activation = activation.reshape((B, H, W, C))
        x = x + activation
        return x

class UpsampleBlock(nn.Module):
    """Two unpadded convolutions and upsample.
    Attributes:
        features: Num convolutional features.
        act_fn: activiation function .
        padding: Default type of padding: 'SAME'.
        use_batch_norm: Whether to use batchnorm at the end or not.
    """

    features: int
    act_fn: callable = nn.tanh
    padding: str = 'SAME'
    use_batch_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, residual, *, train: bool) -> jnp.ndarray:
        x = DeConv3x3(features=self.features, name='deconv', padding=self.padding, use_batch_norm=self.use_batch_norm)(x, train=train)
        if residual is not None:
            x = jnp.concatenate([x, residual], axis=-1)
        x = ConvLayer2(self.features, act_fn=self.act_fn, padding=self.padding, use_batch_norm=self.use_batch_norm)(x, train=train)
        return x

class OutputBlock(nn.Module):
    features: int
    act_fn: callable = nn.tanh
    padding: str = 'SAME'
    use_batch_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        # x = ConvLayer2(self.features, act_fn=self.act_fn, padding=self.padding, 
        #                use_batch_norm=self.use_batch_norm)(x, train=train)
        if self.padding == 'REPLICATE':
            x = Conv_repl(features=self.features, kernel_size=(3, 3), name='conv1x1_repl')(x)
        else:
            x = nn.Conv(features=self.features, kernel_size=(1, 1), name='conv1x1_'+self.padding, padding=self.padding)(x)
        return x


class UNet(nn.Module):
    """
    Attributes:
        act_fn: activiation function
        block_size: Sequence of feature sizes used in UNet blocks.
        padding: Type of padding.
        use_batch_norm: Whether to use batchnorm or not.
    """
    act_fn_name: str
    act_fn: Any
    model_type: str
    block_size: Tuple[int, ...] = (222, 222, 222, 222, 222, 222, 128)
    # block_size: Tuple[int, ...] = (8, 16, 32)
    layers_num: Tuple[int, ...] = (1024, 1024)
    padding: str = 'REPLICATE'
    out_features: int = 222
    use_batch_norm: bool = True
    f_cori: float = 0
    Nc_uv: int = 0
    @nn.compact
    def __call__(self, x: jnp.ndarray, dt, train: bool) -> jnp.ndarray:
        x_input = x
        print(f"The shape of input: {x.shape}")
        B, H, W, C = x.shape
        f_cori = jnp.tile(self.f_cori, (B, 1, 1, self.Nc_uv))
        f_cori_uv = jnp.concatenate((f_cori, -f_cori), axis=-1)
        print(f"The shape of x_input {x.shape} and the shape of f_cori_uv {f_cori_uv.shape}")
        x_input = jnp.concatenate((x_input[...,:2*self.Nc_uv]*(1+dt*f_cori_uv), x_input[...,2*self.Nc_uv:]), axis=-1)
        skip_connections = []
        for i, features in enumerate(self.block_size):
            x, residual = DownsampleBlock(features=features, act_fn=self.act_fn, padding=self.padding, use_batch_norm=self.use_batch_norm, 
                                          name=f'0_down_{i}')(x, train=train)
            skip_connections.append(residual)
            # print(f'{i}, the shape of x: {x.shape}, the shape of residual: {residual.shape}')
        if self.model_type == "U_net":
            x = BottleneckBlock(features=2 * self.block_size[-1], layers_num=self.layers_num, act_fn=self.act_fn, padding=self.padding, use_batch_norm=self.use_batch_norm,
                            name='1_bottleneck')(x, train=train)
        elif self.model_type == "U_net_modified":
            x = BottleneckBlock_modified(features=2 * self.block_size[-1], layers_num=self.layers_num, act_fn=self.act_fn, padding=self.padding, use_batch_norm=self.use_batch_norm,
                            name='1_bottleneck')(x, train=train)
        # print(f'{i}, the shape of x: {x.shape}.')
        upscaling_features = self.block_size[::-1]
        for i, features in enumerate(upscaling_features):
            x = UpsampleBlock(features=features, act_fn=self.act_fn, padding=self.padding, use_batch_norm=self.use_batch_norm,
                              name=f'2_up_{i}')(x, residual=skip_connections.pop(), train=train)
            # print(f'{i}, the shape of x: {x.shape}.')
        x = OutputBlock(features=self.out_features, padding=self.padding, use_batch_norm=self.use_batch_norm, 
                        name='output_projection')(x, train=train)
        return x_input[...,:-2] + dt*x



# x = np.random.normal(size=[1,512,384,222])
# dt = 15*60
# init_key = jax.random.PRNGKey(0)
# mc_model = UNet(act_fn_name="relu", act_fn=nn.relu, model_type="U_net_modified", padding="REPLICATE")
# y, variables = mc_model.init_with_output(init_key, x, dt, train=True)  # initialize via init
# print(y.shape)
# params, batch_stats = variables['params'], variables['batch_stats']