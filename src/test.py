# %%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
# %%
x_min = 0.5
x_max = 12
dx = np.arange(x_min, x_max+0.1, 1)
y = x_max*(1-x_min/dx)/(x_max-x_min)
plt.plot(dx,y)
# %%
flnm = '/home/fengx20/project/sod/data_combine/rain_station_obs_henan.nc'
ds = xr.open_dataset(flnm)
ds
# %%
da = ds['__xarray_dataarray_variable__'].sel(time=slice('2021-07-20 01', '2021-07-21 00')).sum(dim='time')
da.max()