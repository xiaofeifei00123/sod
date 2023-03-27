#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
降水的处理，包括计算总降水，插值等
-----------------------------------------
Time             :2023/03/01 15:19:36
Author           :Forxd
Version          :1.0
'''

# %%
import xarray as xr
import pandas as pd
import numpy as np
import wrf
import netCDF4 as nc
from baobao.interp import regrid_xesmf
import common as com

# %%
def grid2station(flnm_obs, flnm_wrf, flnm_wrfout):
    """将格点数据插值为站点数据

    Args:
        flnm_obs ([type]): 多时次聚合的站点数据
        flnm_wrf ([type]): 多时次聚合的wrfout数据, 某个变量的，例如RAINNC
        flnm_wrfout ([type]): 原始的wrfout数据, 只用作取投影方式的作用

    Returns:
        da_station: 从wrf数据插值出来的和观测经纬度一致的格点数据
    """    

    print("插值数据到站点")
    # 特定区域范围内观测的站点数据
    da_obs = xr.open_dataarray(flnm_obs)
    # wrf输出后聚合到一起的多时间维度数据
    da_wrf = xr.open_dataarray(flnm_wrf)

    cc = da_obs.isel(time=0)
    # sta = cc.id.values
    wrfin = nc.Dataset(flnm_wrfout)
    x, y = wrf.ll_to_xy(wrfin, cc.lat, cc.lon)
    print(y)
    da_station = da_wrf.sel(south_north=y, west_east=x)
    # print(da_station)
    da_station = da_station.drop_vars(['latlon_coord'])
    da_station = da_station.rename({'idx':'sta'})
    # print(da_station)

    return da_station

def regrid_latlon(flnm_rain, area):
    """
    将combine得到的数据()，
    插值到格距较粗的latlon格点上, 
    也包含有投影转换的需求
    插值到latlon格点上
    将二维的latlon坐标水平插值到一维的latlon坐标上
    """
    print("插值数据到网格点")
    ds = xr.open_dataset(flnm_rain)
    ds_out = regrid_xesmf(ds, area)
    return ds_out
    

def rain_station2grid(da):
    """将站点数据，插值为格点数据

    Args:
        da ([type]): [description]

    Returns:
        [type]: [description]
    """
    area = {
        'lon1':110.5,
        'lon2':116,
        'lat1':32,
        'lat2':36.5,
        # 'interval':0.125,
        'interval':0.05,
    }
    ddc = rain_station2grid(da, area)
    return ddc    

def caculate_area_mean_latlon(da,area):
    """计算区域平均降水
    针对维度是latlon的数组
    """
    mask = (
        (da.coords['lat']>area['lat1'])
        &(da.coords['lat']<area['lat2'])
        &(da.coords['lon']<area['lon2'])
        &(da.coords['lon']>area['lon1'])
    )
    aa = xr.where(mask, 1, np.nan)
    db = da*aa
    dsr = db.mean(dim=['lat', 'lon'])
    return dsr

def caculate_area_mean_lambert(da, area,):
    """
    计算区域平均
    针对维度是south_north, west_east的数组
    """
    lon = da['lon'].values
    lat = da['lat'].values
    #     ## 构建掩膜, 范围内的是1， 不在范围的是nan值
        
    clon = xr.where((lon<area['lon2']) & (lon>area['lon1']), 1, np.nan)
    clat = xr.where((lat<area['lat2']) & (lat>area['lat1']), 1, np.nan)
    da = da*clon*clat
    # if 'south_north' in list(da.dims):
    da_mean = da.mean(dim=['south_north', 'west_east'])
    da_mean
    return da_mean

def caculate_sum_time():
    """计算多个时次降水的和，
    因为观测和模式的格点差异所以不进行合并
    分布图不进行插值
    """

    flnm = '/home/fengx20/project/sod/data_combine/rain_all.nc'
    flnm_out = '/home/fengx20/project/sod/data_caculate/rain_sum24h_model.nc'
    ds = xr.open_dataset(flnm)
    ds = ds.sel(time=slice('2021-07-20 01', '2021-07-21 00')).sum(dim='time')
    ds.to_netcdf(flnm_out)
    flnm_obs = '/home/fengx20/project/sod/data_combine/rain_obs.nc'
    flnm_obs_out = '/home/fengx20/project/sod/data_caculate/rain_sum24h_obs.nc'
    ds_obs = xr.open_dataset(flnm_obs)
    ds_obs = ds_obs.sel(time=slice('2021-07-20 01', '2021-07-21 00')).sum(dim='time')
    ds_obs.to_netcdf(flnm_obs_out)
    # ds_obs['PRCP'].sum(dim='time').plot()
    # ds

def caculate_time_sequence():
    ## 模式数据区域平均
    flnm_mdoel = '/home/fengx20/project/sod/data_combine/rain_all.nc'
    ds_m = xr.open_dataset(flnm_mdoel)
    ds_m['precip']
    cm = com.Common()
    da_model = caculate_area_mean_lambert(ds_m['precip'],cm.areaB)
    da_model = da_model.resample(time='1H').sum()  # 因为这里的模式数据是10分钟的，所以重采样
    ## 观测数据区域平均
    flnm_obs = '/home/fengx20/project/sod/data_combine/rain_obs.nc'
    ds_obs = xr.open_dataset(flnm_obs)
    ds_obs['PRCP']
    da_obs = caculate_area_mean_latlon(ds_obs['PRCP'],cm.areaE)
    da_obs = da_obs.sel(time=slice('2021-07-19 12', '2021-07-21 00'))
    ## 合并模式和观测数据
    ds = da_model.to_dataset(dim='model')
    ds['OBS'] = da_obs
    flnm_save = '/home/fengx20/project/sod/data_caculate/rain_time_sequence.nc'
    ds.to_netcdf(flnm_save)


# %%
if __name__ == '__main__':
    # caculate_sum_time()
    caculate_time_sequence()
    pass

# %%
