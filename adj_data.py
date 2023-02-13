import netCDF4 as nc
import numpy as np
from scipy.interpolate import griddata
import time
from utilities import data_interpolation, check_nan
import h5py
import jax.numpy as jnp
import datetime
import utilities


eps = 1e-7
int_setup = True

path = '/work/09012/haoli1/ls6/hurricane/hurricane_data/high_reso/'
file_Temp = 'Pot_Temp_HIDA_d02.nc'
file_Pres = 'Pressu_HIDA_d02.nc'
file_U = 'UVel_HIDA_d02.nc'
file_V = 'VVel_HIDA_d02.nc'

Vals = {}


fh = nc.Dataset(path+file_Temp, mode='r')
print(file_Temp, fh.variables.keys())
Pot_Temp = {}
for key in ['x', 'y']:
    Pot_Temp[key] = np.array(fh.variables[key][:], dtype=np.float32)
    print(f'the shape of {key} is {Pot_Temp[key].shape}')


Vals['x'] = Pot_Temp['x'][0]
Vals['y'] = Pot_Temp['y'][0]

x_start = np.max(Pot_Temp['x'][0,:,0]) + eps
x_end = np.min(Pot_Temp['x'][0,:,-1]) - eps
y_start = np.max(Pot_Temp['y'][0,0]) + eps
y_end = np.min(Pot_Temp['y'][0,-1]) - eps

Nx_int = 512
Ny_int = 384
Nz_int = 10
xs = np.linspace(x_start, x_end, Nx_int, dtype=np.float32)
ys = np.linspace(y_start, y_end, Ny_int, dtype=np.float32)
xx, yy = np.meshgrid(xs, ys)


DATASET_PATH = '/work/09012/haoli1/ls6/hurricane/hurricane_data/high_reso/'
train_file_name = DATASET_PATH+'Nt_313_55x552x669_uvdp_int_2023-02-05.h5'
hf_data = h5py.File(train_file_name, 'r')

Vals = {}
Vals['time'] = np.array(hf_data['time'][:])

Vals['U'] = np.array(hf_data['U'], dtype=np.float32)
print(f"The shape of U {Vals['U'].shape}")
Vals['V'] = np.array(hf_data['V'], dtype=np.float32)
print(f"The shape of V {Vals['V'].shape}")
Vals['D'] = np.array(hf_data['D'], dtype=np.float32)
Vals['P'] = np.array(hf_data['P'], dtype=np.float32)
Vals['x'] = xx
Vals['y'] = yy

Nx = 512
Ny = 384
Nz = 10
Nt = Vals['time'].size

x = datetime.datetime.now()
if int_setup:
    file_name = 'Nt_' +str(Nt) + '_' + str(Nz) + 'x' + str(Nx) + 'x' + str(Ny) + "_" + 'uvdp_int_'+ str(x)[:10] + '.h5'
else:
    file_name = 'Nt_' +str(Nt) + '_' + str(Nz) + 'x' + str(Nx) + 'x' + str(Ny) + "_" + 'uvdp_' + str(x)[:10] + '.h5'
fh_int = h5py.File(path+file_name, 'w')

for key in Vals.keys():
    fh_int.create_dataset(key, data=Vals[key])
    print(f'the shape of {key} is {Vals[key].shape}')