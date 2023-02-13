from scipy.interpolate import griddata
import time
import numpy as np
import jax.numpy as jnp 

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

def norm_data(h5_file, Nz, dim_set='3d'):
    norm_paras = {}
    T = np.array(h5_file['time'][:])
    Nt = T.size
    
    D = np.array(h5_file['D'][..., :Nz], dtype=np.float32)
    Dmax = np.max(D)
    Dmin = np.min(D)
    D = (D-Dmin)/(Dmax-Dmin)
    norm_paras['D'] = (Dmin, Dmax)

    P = np.array(h5_file['P'][..., :Nz], dtype=np.float32)
    Pmax = np.max(P)
    Pmin = np.min(P)
    P = (P-Pmin)/(Pmax-Pmin)
    norm_paras['P'] = (Pmin, Pmax)

    U = np.array(h5_file['U'][..., :Nz], dtype=np.float32)
    Umax = np.max(U)
    Umin = np.min(U)
    U = (U-Umin)/(Umax-Umin)
    norm_paras['U'] = (Umin, Umax)

    V = np.array(h5_file['V'][..., :Nz], dtype=np.float32)
    Vmax = np.max(V)
    Vmin = np.min(V)
    V = (V-Vmin)/(Vmax-Vmin)
    norm_paras['V'] = (Vmin, Vmax)

    x = np.array(h5_file['x'][:])
    xmax = np.max(x)
    xmin = np.min(x)
    x = (x-xmin)/(xmax-xmin)
    norm_paras['x'] = (xmin, xmax)

    y = np.array(h5_file['y'][:])
    ymax = np.max(y)
    ymin = np.min(y)
    y = (y-ymin)/(ymax-ymin)
    norm_paras['y'] = (ymin, ymax)

    if dim_set == '3d':
        xx = np.tile(np.expand_dims(x, axis=(0,-1)), (Nt,1,1,1))
        yy = np.tile(np.expand_dims(y, axis=(0,-1)), (Nt,1,1,1))
        primes_data = np.concatenate([U, V, D, P, xx, yy], axis=-1)
        return primes_data, norm_paras, xx[0], yy[0]
    elif dim_set == '2d':
        xx = np.tile(np.expand_dims(x, axis=(0,-1)), (Nt,1,1,Nz))
        yy = np.tile(np.expand_dims(y, axis=(0,-1)), (Nt,1,1,Nz))
        primes_data = np.stack([U, V, D, P, xx, yy], axis=-1)
        primes_data = jnp.moveaxis(primes_data, -2, 0)
        return primes_data, norm_paras, xx[0,:,:,:1], yy[0,:,:,:1]
    
def recover_data(norm_paras, reco_list, norm_data_list):
    for i in range(len(reco_list)):
        key = reco_list[i]
        datamin, datamax = norm_paras[key]
        norm_data_list[i] = (datamax-datamin)*norm_data_list[i] + datamin
    return norm_data_list
