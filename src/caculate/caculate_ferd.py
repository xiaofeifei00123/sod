#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
根据wrfout计算弗劳德数Fr
-----------------------------------------
Time             :2023/03/06 20:18:01
Author           :Forxd
Version          :1.0
'''
# %%
import os
import xarray as xr
import pandas as pd
import numpy as np
import wrf
import netCDF4 as nc
from metpy.calc import brunt_vaisala_frequency_squared,second_derivative,first_derivative
from metpy.calc import *
from metpy.units import units
import matplotlib.pyplot as plt
from baobao.timedur import timeit
# %%
@timeit
def caculate_fr(flnm):
    """ 
    flnm: wrfout的路径
    计算弗劳德数
    """
    wrfin = nc.Dataset(flnm)
    ds = xr.Dataset()
    ds['u'] = wrf.getvar(wrfin, 'ua')
    ds['v'] = wrf.getvar(wrfin, 'va')
    ds['z'] = wrf.getvar(wrfin, 'z')
    ds['temp'] = wrf.getvar(wrfin, 'temp', units='degC')
    ds['td'] = wrf.getvar(wrfin, 'td', units='degC')
    ds['pressure'] = wrf.getvar(wrfin, 'pressure')
    pressure = units.Quantity(ds.pressure.values, "hPa")
    dew_point = units.Quantity(ds['td'].values, "degC")
    temperature = units.Quantity(ds['temp'].values, "degC")

    q = specific_humidity_from_dewpoint(pressure, dew_point)
    w = mixing_ratio_from_specific_humidity(q)
    theta_v = virtual_potential_temperature(pressure, temperature, w)
    height = units.Quantity(ds['z'].values, "m")
    N2 = brunt_vaisala_frequency_squared(height,theta_v.squeeze(), vertical_dim=0)
    wind_speed = np.sqrt(ds['u']**2+ds['v']**2)
    Fr = wind_speed/np.sqrt(N2)/1000

    da = xr.DataArray(
                Fr.values,
                coords = ds['u'].coords,
                dims = ds['u'].dims,
            )
    da.name = 'fr'
    dss = da.to_dataset()
    return dss

@timeit
def write_xarray_to_netcdf(xarray_array, output_path,mode='w', format='NETCDF4', group=None, engine=None,
                           encoding=None):
    """删除变量的attrs中的cooordinates和projection"""
    if 'dataarray' in str(type(xarray_array)):
        if 'projection' in xarray_array.attrs:
            del xarray_array[var].attrs['projection']
        if 'coordinates' in xarray_array.attrs:
            del xarray_array.attrs['coordinates']

        xarray_array.to_netcdf(path=output_path, mode=mode, format=format, group=group,
                                engine=engine,
                                encoding=encoding)
                            
    elif 'dataset' in str(type(xarray_array)):

        for var in list(xarray_array.data_vars):
            if 'coordinates' in xarray_array.attrs:
                del xarray_array[var].attrs['coordinates']
            # if xarray_array[var].attrs['projection']:
            if 'projection' in xarray_array.attrs:
                del xarray_array[var].attrs['projection']

        xarray_array.to_netcdf(path=output_path, mode=mode, format=format, group=group,
                                engine=engine,
                                encoding=encoding)

def combine_time():
    pass

    model_list = ['sod_all', 'sod_bl', 'sod_fd', 'sod_ss', 'sod_ss', 'sod_no']
    for model in model_list:
        print(model)
        path = '/home/fengx20/project/sod/data_original/wrfout/'+model+'/'
        # fl_list = os.popen('ls {}/wrfout_d02*'.format(path))  # 打开一个管道
        # fl_list = os.popen('ls {}/wrfout/wrfout_d03*'.format(path))  # 打开一个管道
        # fl_list = fl_list.read().split()
        tt = pd.date_range('2021-07-19 20', '2021-07-20 08', freq='2H')
        fl_list = []
        for t in tt:
            str_time= t.strftime('%Y-%m-%d_%H:%M:%S')
            fl = 'wrfout_d03_'+str_time
            fl_list.append(fl)


        dss_list = []
        for fl in fl_list:
            print(fl[-20:])
            flnm = path+fl
            dss = caculate_fr(flnm)
            dss_list.append(dss)
        ds2 = xr.concat(dss_list, dim='Time')
        # ds2.to_netcdf(path+'/fr.nc')
        write_xarray_to_netcdf(ds2, path+'/fr.nc')

def combine_model():
    path = '/home/fengx20/project/sod/data_original/wrfout/'
    ds_list = []
    model_list = ['sod_all', 'sod_bl', 'sod_fd', 'sod_ss', 'sod_ss', 'sod_no']
    for model in model_list:
        path_model = path+'/'+model+'/fr.nc'
        ds = xr.open_dataset(path_model)
        ds_list.append(ds)
    
    path_to = '/home/fengx20/project/sod/data_caculate/'
    ds2 = xr.concat(ds_list,pd.Index(model_list,name='model'))
    ds2.to_netcdf(path_to+'/fr.nc')
    pass

if __name__ == "__main__":
    # flnm_wrf = '../../data_original/wrfout/sod_all/wrfout_d03_2021-07-19_15:20:00'
    # dss = caculate_fr(flnm_wrf)
    # flnm_save = '/home/fengx20/project/sod/data_caculate/Fr.nc'
    # write_xarray_to_netcdf(dss, flnm_save)
    combine_time()
    combine_model()
# %%
flnm = '../../data_original/wrfout/sod_all/wrfout_d03_2021-07-19_15:20:00'
wrfin = nc.Dataset(flnm)
ds = xr.Dataset()
ds['u'] = wrf.getvar(wrfin, 'ua')
ds['v'] = wrf.getvar(wrfin, 'va')
ds['z'] = wrf.getvar(wrfin, 'z')
ds['temp'] = wrf.getvar(wrfin, 'temp', units='degC')
ds['td'] = wrf.getvar(wrfin, 'td', units='degC')
ds['pressure'] = wrf.getvar(wrfin, 'pressure')
pressure = units.Quantity(ds.pressure.values, "hPa")
dew_point = units.Quantity(ds['td'].values, "degC")
temperature = units.Quantity(ds['temp'].values, "degC")

q = specific_humidity_from_dewpoint(pressure, dew_point)
w = mixing_ratio_from_specific_humidity(q)
theta_v = virtual_potential_temperature(pressure, temperature, w)
height = units.Quantity(ds['z'].values, "m")
N2 = brunt_vaisala_frequency_squared(height,theta_v.squeeze(), vertical_dim=0)
wind_speed = np.sqrt(ds['u']**2+ds['v']**2)
Fr = wind_speed/np.sqrt(N2)/1000