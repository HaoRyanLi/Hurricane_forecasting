from scipy.interpolate import griddata
import time
import numpy as np
import jax.numpy as jnp 
import matplotlib.pyplot as plt
import matplotlib

import tkinter
import matplotlib
matplotlib.use('TkAgg')


def data_interpolation(data_orig, Nt, Nz, points, xx, yy):
    data_int = []
    for i in range(Nt):
        values_int = []
        start_time = time.perf_counter()
        for j in range(Nz):
            values = data_orig[i][j].reshape((-1,))
            val = griddata(points, values, (xx, yy), method='linear')
            values_int.append(val)
        end_time = time.perf_counter()
        print(f"The {i}-th interpolation loop takes {end_time-start_time}.")
        data_int.append(np.stack(values_int, axis=-1))
        del values_int
    return np.stack(data_int, axis=0)


def h5file_to_data(h5_file, Nz=55):
    '''
    Now consider all the data structure as 2D and all the vars on z-direction as features.
    '''
    T = np.array(h5_file['time'][:])
    Nt = T.size
    U = np.array(h5_file['U'][:, :Nz], dtype=np.float32)
    V = np.array(h5_file['V'][:, :Nz], dtype=np.float32)
    D = np.array(h5_file['D'][:, :Nz], dtype=np.float32)
    P = np.array(h5_file['P'][:, :Nz], dtype=np.float32)
    x = np.array(h5_file['x'][:])
    y = np.array(h5_file['y'][:])
    xx = np.tile(np.expand_dims(x, axis=(0,-1)), (Nt,1,1,1))
    print(f"The shape of xx {xx.shape}")
    yy = np.tile(np.expand_dims(y, axis=(0,-1)), (Nt,1,1,1))
    primes_data = np.concatenate([U, V, D, P, xx, yy], axis=-3)
    print(f"The shape of primes_data {primes_data.shape}")
    return np.moveaxis(primes_data, -3, -1)

def compare_data(data0, data1, keys):
    for key in keys:
        print(key, np.sum((data0[key][:] - data1[key][:])**2))

def check_nan(data_arr):
    return np.sum(np.isnan(data_arr))

def rolling_window(a: jnp.ndarray, window: int):
    idx = jnp.arange(len(a) - window + 1)[:, None] + jnp.arange(window)[None, :]
    return a[idx]

# Transform data to disired shape: (n_train_samples, Nt, Nx) -> (n_train_samples, Nt-n_seq+1, n_seq, Nx)
def transform_data(data, n_seq):
    batched_samples = rolling_window(data, n_seq)
    return batched_samples

def norm_data(h5_file, norm_paras, Nz, dim_set='3d'):
    T = np.array(h5_file['time'][:])
    Nt = T.size
    
    D = np.array(h5_file['D'][..., :Nz], dtype=np.float32)
    Dmin, Dmax = norm_paras['D']
    D = (D-Dmin)/(Dmax-Dmin)

    P = np.array(h5_file['P'][..., :Nz], dtype=np.float32)
    Pmin, Pmax = norm_paras['P']
    P = (P-Pmin)/(Pmax-Pmin)

    U = np.array(h5_file['U'][..., :Nz], dtype=np.float32)
    Umin, Umax = norm_paras['U']
    U = (U-Umin)/(Umax-Umin)

    V = np.array(h5_file['V'][..., :Nz], dtype=np.float32)
    Vmin, Vmax = norm_paras['V']
    V = (V-Vmin)/(Vmax-Vmin)

    x = np.array(h5_file['x'][:])
    xmin, xmax = norm_paras['x']
    x = (x-xmin)/(xmax-xmin)

    y = np.array(h5_file['y'][:])
    ymin, ymax = norm_paras['y']
    y = (y-ymin)/(ymax-ymin)

    if dim_set == '3d':
        xx = np.tile(np.expand_dims(x, axis=(0,-1)), (Nt,1,1,1))
        yy = np.tile(np.expand_dims(y, axis=(0,-1)), (Nt,1,1,1))
        primes_data = np.concatenate([U, V, D, P, xx, yy], axis=-1)
        # primes_data = np.expand_dims(primes_data, axis=0)
        return primes_data, xx[0], yy[0]
    elif dim_set == '2d':
        xx = np.tile(np.expand_dims(x, axis=(0,-1)), (Nt,1,1,Nz))
        yy = np.tile(np.expand_dims(y, axis=(0,-1)), (Nt,1,1,Nz))
        primes_data = np.stack([U, V, D, P, xx, yy], axis=-1)
        primes_data = jnp.moveaxis(primes_data, -2, 0)
        return primes_data, xx[0,:,:,:1], yy[0,:,:,:1]
    
def recover_norm_data(norm_paras, key_list, norm_data_list):
    data_list = []
    for i in range(len(key_list)):
        key = key_list[i]
        datamin, datamax = norm_paras[key]
        print(key, datamin, datamax)
        data_list.append((datamax-datamin)*norm_data_list[i] + datamin)
    return data_list

def recover_norm_data_woxy(norm_paras, norm_data, Nz=1):
    '''
    norm_data: normalized data array with shape [B, H, W, 4*Nz]. the data sequence of norm_data should be [U, V, D, P]
    Nz: the num. of z levels of the data.
    '''
    U, V, D, P = norm_data[...,:Nz], norm_data[...,Nz:2*Nz], norm_data[...,2*Nz:3*Nz], norm_data[...,3*Nz:4*Nz]
    norm_data_list = [U, V, D, P]
    key_list = ['U', 'V', 'D', 'P']
    data = recover_norm_data(norm_paras, key_list, norm_data_list)
    return jnp.concatenate(data, axis=-1)

def recover_norm_data_wxy(norm_paras, norm_data, Nz=1):
    '''
    norm_data: normalized data array with shape [B, H, W, 4*Nz+2]. the data sequence of norm_data should be [U, V, D, P]
    Nz: the num. of z levels of the data.
    '''
    U, V, D, P, xx, yy = norm_data[...,:Nz], norm_data[...,Nz:2*Nz], norm_data[...,2*Nz:3*Nz], norm_data[...,3*Nz:4*Nz]
    norm_data_list = [U, V, D, P, xx, yy]
    key_list = ['U', 'V', 'D', 'P', 'x', 'y']
    data = recover_norm_data(norm_paras, key_list, norm_data_list)
    return jnp.concatenate(data, axis=-1)

def plot_fig(Z, title, Zmin, Zmax, x_start, x_end, y_start, y_end, save_imag_path, plot_show):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    im = ax.imshow(Z, extent=(x_start,x_end, y_start, y_end), cmap=matplotlib.cm.RdBu, vmin=Zmin, vmax=Zmax)
    cb = fig.colorbar(im, ax=ax)
    fig.savefig(save_imag_path, format='eps')
    if plot_show:
        plt.show()

def get_max_min(data, Nz):
    keys = ['U', 'V', 'D', 'P']
    U = data[...,:Nz]
    V = data[...,Nz:2*Nz]
    D = data[...,2*Nz:3*Nz]
    P = data[...,3*Nz:4*Nz]
    data_list = [U, V, D, P]
    for i in range(4):
        key = keys[i]
        print(f"The max value of {key} is {np.max(data_list[i])}. The min value of {key} is {np.min(data_list[i])}")

def get_real_rel_err(norm_paras, norm_data_nn, norm_data_true, Nz=1):
    data_nn = recover_norm_data_woxy(norm_paras, norm_data_nn, Nz)
    data_true = recover_norm_data_woxy(norm_paras, norm_data_true, Nz)
    keys = ['U', 'V', 'D', 'P']
    rel_err_sum = 0
    err_list = []
    for i in range(4):
        key = keys[i]
        print(f"The shape of data {key} is {data_true[...,i*Nz:(i+1)*Nz].shape}")
        rel_err = jnp.mean((data_nn[...,i*Nz:(i+1)*Nz]-data_true[...,i*Nz:(i+1)*Nz])**2)/jnp.mean(data_true[...,i*Nz:(i+1)*Nz]**2)
        rel_err_sum +=  rel_err
        err_list.append(rel_err)
    return *err_list, rel_err_sum

def get_weighted_loss(err, scal_fact, Nz):
    err_u, err_v, err_d, err_p = jnp.mean(err[...,:Nz]), jnp.mean(err[...,Nz:2*Nz]), jnp.mean(err[...,2*Nz:3*Nz]), jnp.mean(err[...,3*Nz:4*Nz])
    err_arr = jnp.array([err_u, err_v, err_d, err_p])
    return jnp.dot(scal_fact, err_arr)/jnp.sum(scal_fact)