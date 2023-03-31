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
# from metpy.calc import *
import metpy.calc as cal
from metpy.units import units
import matplotlib.pyplot as plt
from baobao.timedur import timeit
import metpy.constants as ms
import atmos
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

    q = cal.specific_humidity_from_dewpoint(pressure, dew_point)
    w = cal.mixing_ratio_from_specific_humidity(q)
    theta_v = cal.virtual_potential_temperature(pressure, temperature, w)
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
    ds2 = xr.concat(ds_list,pd.Index(model_list, name='model'))
    ds2.to_netcdf(path_to+'/fr.nc')
    pass

if __name__ == "__main__":
    # flnm_wrf = '../../data_original/wrfout/sod_all/wrfout_d03_2021-07-19_15:20:00'
    # dss = caculate_fr(flnm_wrf)
    # flnm_save = '/home/fengx20/project/sod/data_caculate/Fr.nc'
    # write_xarray_to_netcdf(dss, flnm_save)

    pass

    # combine_time()
    # combine_model()
# %%
# flnm = '../../data_original/wrfout/sod_all/wrfout_d03_2021-07-19_15:20:00'
flnm = '../../data_original/wrfout/sod_all/wrfout_d03_2021-07-20_04:20:00'
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

q = cal.specific_humidity_from_dewpoint(pressure, dew_point)
w = cal.mixing_ratio_from_specific_humidity(q)
theta_v = cal.virtual_potential_temperature(pressure, temperature, w)
height = units.Quantity(ds['z'].values, "m")
N2 = brunt_vaisala_frequency_squared(height,theta_v.squeeze(), vertical_dim=0)
wind_speed = np.sqrt(ds['u']**2+ds['v']**2)
Fr = wind_speed/np.sqrt(N2)/1000
# %%
# Fr
# cal
# cal.relative_humidity_from_dewpoint(pressure,dewpoint)
# pressure.load()
# p = pressure.magnitude
# p
# temperature
# ds['pressure']
# %%
# p

# pressure = units.Quantity(ds.pressure.values, "hPa")
# ds
# pressure = units.Quantity(ds.pressure.values, "hPa")
# Fr
# cal.relative_humidity_from_dewpoint(pressure,dew_point)
rh = cal.relative_humidity_from_specific_humidity(pressure,temperature,q)
# %%
# pass
# rh.magnitude
da = xr.DataArray(rh[:,:,2].magnitude)
da.plot(levels=[0.8,0.9,0.95,1])
# ds['u']
# ds.Time
# %%
T = ds['temp'].values+273.15
T
# %%
g = ms.earth_gravity.magnitude
L = ms.water_heat_vaporization.magnitude  # 潜热, L
Rv = ms.water_gas_constant.magnitude  # 气体常数, Rv
R = ms.dry_air_gas_constant.magnitude# 气体常数, Rv
cp = ms.wv_specific_heat_press.magnitude
# %%
es = atmos.equations.es_from_T_Bolton(ds['temp']+273.15)
qs = atmos.equations.rvs_from_p_es(ds['pressure'], es)
rv = qs
rt = atmos.equations.rt_from_rv(rv)
qw = rt
xi = R/Rv
# %%
term1 = 1+L*qs/R*T
# L
term2 = 1+xi*(L**2)*qs/(cp*R*T**2)

# %%
theta = atmos.equations.theta_from_p_T(ds['pressure'], ds['temp'])
# %%
# term3 = 1/theta* 
z = ds['z'].values
# z.shape
# theta.shape

term31 = (1/theta*cal.first_derivative(theta, axis=0, x=z))
term32 = (cal.first_derivative(qs,axis=0,x=z))
term3 = term31*L/cp/T*term32
# z.dims
# %%
term4 = cal.first_derivative(qw,axis=0,x=z)
Nm2 = g*((term1/term2)*term3-term4)
# %%
# Nm2
u = ds['u']
v = ds['v']
dudz = first_derivative(u,axis=0,x=z)
dvdz = first_derivative(v,axis=0,x=z)
# %%
# Ri = N2/(dudz**2+dvdz**2)
# dudz
Nm2 = Nm2.magnitude
# %%
dudz = dudz.magnitude
dvdz = dvdz.magnitude
# %%
Ri = Nm2/(dudz**2+dvdz**2)
Ri
# N2
# %%
# Nm2
# Ri
# dudz
# xr.DataArray(Ri)[1,:,:].plot()
# Ri.min()
# %%
lon = ds.XLONG.values
lat = ds.XLAT.values
# xr.DataArray(Ri)[1,:,:].plot(levels=[-1, 0, 0.1, 0.5, 1, 2, 10, 20, 100, 200])

# %%
da =  xr.DataArray(
        Ri,
        coords={
            'lon':(('south_north', 'west_east'),lon),
            'lat':(('south_north', 'west_east'),lat),
            # 'bottom_top':(('bottom_top','south_north', 'west_east'),lat),

        },
        dims =['bottom_top','south_north', 'west_east']
    )

# %%
# ds['u'].shape
da



# %%
def caculate_average_wrf(da, area = {'lat1':33.5, 'lat2':34, 'lon1':112, 'lon2':113,}):
    """求wrfout数据，在区域area内的区域平均值

    Args:
        da ([type]): 直接利用wrf-python 从wrfout中得到的某个变量的数据

        area = {
            'lat1':33,
            'lat2':34,
            'lon1':111.5,
            'lon2':113,
        }
    Returns:
        [type]: [description]
    """
    lon = da['XLONG'].values
    lat = da['XLAT'].values
    # lon = da['lon'].values
    # lat = da['lat'].values
    ## 构建掩膜, 范围内的是1， 不在范围的是nan值
    clon = xr.where((lon<area['lon2']) & (lon>area['lon1']), 1, np.nan)
    clat = xr.where((lat<area['lat2']) & (lat>area['lat1']), 1, np.nan)
    da = da*clon*clat
    da_mean = da.mean(dim=['south_north', 'west_east'])
    return da_mean

# %%
db = caculate_average_wrf(da)
# %%
# db.plot()

# %%
# db.bottom_top
# z
zz = caculate_average_wrf(ds['z'])
zz
# %%
fig = plt.figure(figsize=(4,3),dpi=300)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(db,zz)
# ax.set_ylim(0, 4000)
# ax.set_xlim(-1000,1000)
# print(1)
# ds['z']