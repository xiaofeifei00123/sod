#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:

-----------------------------------------
Time             :2023/03/29 17:59:59
Author           :Forxd
Version          :1.0
'''


# %%
import xarray as xr
import caculate_Ri as cr

# %%
flnm = '/home/fengx20/project/sod/data_combine/micaps_sounding_station_all.nc'
ds = xr.open_dataset(flnm)
ds1 = ds.sel(station='nanyang').isel(time=10)
ds2 = ds1.dropna(dim='vertical')
ds2 = ds2.rename({'height':'z'})
# %%
Nm_obs = cr.caculate_Nm(ds2)
scorer, scorerT = cr.caculate_scorer(Nm_obs,ds2)
flnm = '../../data_original/wrfout/sod_all/wrfout_d03_2021-07-20_00:20:00'
hor_wind = cr.caculate_cross_wind(flnm, ds2)
# %%