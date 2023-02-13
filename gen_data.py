import netCDF4 as nc
import numpy as np
from scipy.interpolate import griddata
import time
from utilities import data_interpolation, check_nan
import h5py
import jax.numpy as jnp
import datetime

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
for key in fh.variables.keys():
    Pot_Temp[key] = np.array(fh.variables[key][:], dtype=np.float32)
    print(f'the shape of {key} is {Pot_Temp[key].shape}')

Pot_Temp['T'] = Pot_Temp['T']+300

Vals['x'] = Pot_Temp['x'][0]
Vals['y'] = Pot_Temp['y'][0]
Vals['z'] = Pot_Temp['z'][0]
Vals['time'] = Pot_Temp['time']

x_start = np.max(Pot_Temp['x'][0,:,0]) + eps
x_end = np.min(Pot_Temp['x'][0,:,-1]) - eps
y_start = np.max(Pot_Temp['y'][0,0]) + eps
y_end = np.min(Pot_Temp['y'][0,-1]) - eps
z_end = np.min(Pot_Temp['z'][0])  + eps # no error as z_end~0 < z_start~1
z_start = np.max(Pot_Temp['z'][0]) - eps

Nx_int = 512
Ny_int = 384
Nz_int = 10
xs = np.linspace(x_start, x_end, Nx_int, dtype=np.float32)
ys = np.linspace(y_start, y_end, Ny_int, dtype=np.float32)
zs = np.linspace(z_start, z_end, Nz_int, dtype=np.float32)
xx, yy = np.meshgrid(xs, ys)

Nz, Nx, Ny = Pot_Temp['T'][0].shape
x_orig = np.expand_dims(Pot_Temp['x'][0], axis=0).reshape((-1,))
y_orig = np.expand_dims(Pot_Temp['y'][0], axis=0).reshape((-1,))
points_orig = np.stack((x_orig, y_orig), axis=-1)


Nt = Pot_Temp['time'].size
x = datetime.datetime.now()
if int_setup:
    file_name = 'Nt_' +str(Nt) + '_' + str(Nz) + 'x' + str(Nx) + 'x' + str(Ny) + "_" + 'uvdp_int_'+ str(x)[:10] + '.h5'
else:
    file_name = 'Nt_' +str(Nt) + '_' + str(Nz) + 'x' + str(Nx) + 'x' + str(Ny) + "_" + 'uvdp_' + str(x)[:10] + '.h5'
fh_int = h5py.File(path+file_name, 'w')


Nz = 10
if int_setup:
    Vals['T'] = data_interpolation(Pot_Temp['T'][:,:Nz], Nt, Nz, points_orig, xx, yy)
    print(f"The shape T after interpolation {Vals['T'].shape}")
    nan_num = check_nan(Vals['T'])
    if nan_num == 0:
        fh_int.create_dataset('T', data=Vals['T'])
    else:
        print(f"The exits {nan_num} Nan values")
else:
    Vals['T'] = Pot_Temp['T'][:,:Nz]
    print(f"The shape T {Vals['T'].shape}")


# values = Pot_Temp['T'][0,0].reshape((-1,))
# points = np.stack((Pot_Temp['x'][0].reshape((-1,)), Pot_Temp['y'][0].reshape((-1,))), axis=-1)
# grid_y, grid_x = np.mgrid[y_start:y_end:120j, x_start:x_end:320j]
# value_z0 = griddata(points, values, (grid_x, grid_y), method='cubic')

# print(f"The difference between two interpolation {np.sum((values_int[0]-value_z0)**2)}, {np.sum((values_int[0]-value_z0)**2)/np.sum((value_z0)**2)}")

del Pot_Temp

fh = nc.Dataset(path+file_Pres, mode='r')
print(file_Pres, fh.variables.keys())
Pres = {}
for key in fh.variables.keys():
    Pres[key] = np.array(fh.variables[key][:], dtype=np.float32)
    print(f'the shape of {key} is {Pres[key].shape}')

if int_setup:
    Vals['P'] = data_interpolation(Pres['P'][:,:Nz], Nt, Nz, points_orig, xx, yy)
    print(f"The shape P after interpolation {Vals['P'].shape}")
    nan_num = check_nan(Vals['P'])
    if nan_num == 0:
        fh_int.create_dataset('P', data=Vals['P'])
    else:
        print(f"The exits {nan_num} Nan values")
else:
    Vals['P'] = Pres['P'][:,:Nz]
    print(f"The shape P {Vals['P'].shape}")



del Pres

R = 287
Vals['D'] = Vals['P']/Vals['T']/R
del Vals['T']
if int_setup:
    nan_num = check_nan(Vals['D'])
    if nan_num == 0:
        fh_int.create_dataset('D', data=Vals['D'])
    else:
        print(f"The exits {nan_num} Nan values")

fh = nc.Dataset(path+file_U, mode='r')
print(file_U, fh.variables.keys())
Vel_U = {}
for key in fh.variables.keys():
    Vel_U[key] = np.array(fh.variables[key][:], dtype=np.float32)
    print(f'the shape of {key} is {Vel_U[key].shape}')

if int_setup:
    Vals['U'] = data_interpolation(Vel_U['U'][:,:Nz], Nt, Nz, points_orig, xx, yy)
    print(f"The shape U after interpolation {Vals['U'].shape}")
    nan_num = check_nan(Vals['U'])
    if nan_num == 0:
        fh_int.create_dataset('U', data=Vals['U'])
    else:
        print(f"The exits {nan_num} Nan values")

else:
    Vals['U'] = Vel_U['U'][:,:Nz]
    print(f"The shape U {Vals['U'].shape}")


del Vel_U


fh = nc.Dataset(path+file_V, mode='r')
print(file_V, fh.variables.keys())
Vel_V = {}
for key in fh.variables.keys():
    Vel_V[key] = np.array(fh.variables[key][:], dtype=np.float32)
    print(f'the shape of {key} is {Vel_V[key].shape}')

if int_setup:
    Vals['V'] = data_interpolation(Vel_V['V'][:,:Nz], Nt, Nz, points_orig, xx, yy)
    print(f"The shape V after interpolation {Vals['V'].shape}")
    
    nan_num = check_nan(Vals['V'])
    if nan_num == 0:
       fh_int.create_dataset('V', data=Vals['V'])
    else:
        print(f"The exits {nan_num} Nan values")

else:
    Vals['V'] = Vel_V['V'][:,:Nz]
    print(f"The shape V {Vals['V'].shape}")


del Vel_V
checkee = list(fh_int.keys())
print(checkee)
for key in Vals.keys():
    if key not in checkee:
        fh_int.create_dataset(key, data=Vals[key])
    print(f'the shape of {key} is {Vals[key].shape}')

# D = jnp.array(Vals['D'][:], dtype=jnp.float32)
# P = jnp.array(Vals['P'][:], dtype=jnp.float32)
# U = jnp.array(Vals['U'][:], dtype=jnp.float32)
# V = jnp.array(Vals['V'][:], dtype=jnp.float32)
# all_data = jnp.concatenate([D, P, U, V], axis=-3)
# print(all_data.shape)


fh_int.close()