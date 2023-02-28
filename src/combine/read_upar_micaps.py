#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
读取高空填图数据(站点)
插值成格点
聚合成一个文件
和wrfout插值出的格点数据作差
这里height表示位势高度
-----------------------------------------
Time             :2021/10/06 16:59:52
Author           :Forxd
Version          :1.0
'''

# %%
from time import strftime
from cartopy.crs import Projection
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
import xarray as xr
import meteva.base as meb
import metpy.interpolate as interp
import numpy as np
import os
import pandas as pd
from nmc_met_io.read_micaps import read_micaps_1, read_micaps_2, read_micaps_14
import meteva.base as meb
from nmc_met_graphics.plot import mapview
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xesmf as xe
from read_global import caculate_diagnostic
from multiprocessing import Pool

from baobao.interp


# %%


def frame_dataset(sta):
    sta = sta.rename(columns={'dewpoint_depression':'t_td', 'temperature':'temp', 'level':'pressure'})
    sta['u'] = -1*np.sin(sta['wind_angle']/180*np.pi)*sta['wind_speed']
    sta['v'] = -1*np.cos(sta['wind_angle']/180*np.pi)*sta['wind_speed']
    sta['td'] = sta['temp'] - sta['t_td']
    sta.index.name = 'station'
    sta.columns.name = 'var'
    ds = xr.Dataset(sta)
    ds = ds.swap_dims({'station':'id'})
    ds = ds.set_coords(['lat', 'lon', 'pressure'])
    diag = caculate_diagnostic(ds)
    ds1 = xr.merge([ds, diag])
    t = ds1['time'][0]
    ds2 = ds1.drop_vars(['dtime', 'grade', 'wind_angle', 'wind_speed', 'time'])
    ds3 = ds2.assign_attrs({'time':t.values})
    return ds3

def get_plot(dic):
    """读取micaps 2类数据，高空填图数据"""
    path = '/mnt/zfm_18T/fengxiang/HeNan/Data/OBS/Micaps/high/PLOT/'
    flnm = path+str(dic['level'])+'/'+dic['time'].strftime('%Y%m%d%H%M%S.000')
    # flnm = '/mnt/zfm_18T/fengxiang/HeNan/Data/OBS/Micaps/high/PLOT/500/20210720080000.000'
    df = read_micaps_2(flnm)
    sta = meb.sta_data(df, columns = [
                'id', 'lon', 'lat', 'alt', 'grade', 'height', 'temperature', 'dewpoint_depression',
                'wind_angle', 'wind_speed', 'time', 'level'])

    return sta


def interp_metpy(da):
    """
    站点插值到格点
    反距离权重插值

    Args:
        sta (DataFrame): [lon,lat,height]

    Returns:
        [type]: [description]
    """
    # h = sta['height']
    # h = sta[var]
    # lon = sta['lon']
    # lat = sta['lat']
    h = da.values
    lon = da.lon.values
    lat = da.lat.values

    area = {
        'lon1':107-1,
        'lon2':135+1,
        'lat1':20-1,
        'lat2':40+1,
        'interval':0.5,
    }

    ds_regrid = xe.util.grid_2d(area['lon1'], area['lon2'], area['interval'], area['lat1'], area['lat2'], area['interval'])
    mx = ds_regrid['lon'].values
    my = ds_regrid['lat'].values

    # z = interp.inverse_distance_to_grid(lon, lat, h, mx, my, r=5, min_neighbors=1)
    z = interp.inverse_distance_to_grid(lon, lat, h, mx, my, r=2, min_neighbors=1)

    ## 重新设置coords属性
    lon1 = mx[0,:]
    lat1 = my[:,0]

    da = xr.DataArray(
        z,
        coords={
            'lon':lon1,
            'lat':lat1,
        },
        dims=['lat','lon']
    )
    return da


def get_one(t, level):
    """获得一个时次, 一个层次的插值过后的高空填图数据
    """
    # t = pd.Timestamp('2021-07-19 2000')
    dic_t = {
        'var':'temp',
        'level':level,
        'time':t
    }
    ## 读取站点数据
    sta = get_plot(dic_t)
    ## 站点数据由DataFrame转为Dataset, 同时计算q,rh, theta_v等变量
    ds = frame_dataset(sta)

    ## 对各个变量分别进行站点插值到格点
    var_list = list(ds.var()) ## 获取Dataset全部变量
    ds_regrid = xr.Dataset()
    for var in var_list:
        da = interp_metpy(ds[var])
        ds_regrid[var] = da
    ds_regrid.assign_attrs({'time':t})
    return ds_regrid

def main():
    pressure_list = [200, 500, 850]
    # ttt = pd.date_range(start='2021-07-18 08', end='2021-07-20 20', freq='12H')
    ttt = pd.date_range(start='2021-07-18 08', end='2021-07-18 20', freq='12H')
    time_list = []
    for t in ttt:
        level_list = []
        for level in ['200', '500', '850']:
            print('读%s'%level)
            ds1 = get_one(t, level)
            level_list.append(ds1)
            # print(a)
        ds2 = xr.concat(level_list, pd.Index(pressure_list, name='pressure'))
        time_list.append(ds2)
    ds3 = xr.concat(time_list, pd.Index(ttt, name='time'))
    return ds3

def multi_one(t):
    pass
    pressure_list = [200, 500, 850]
    level_list = []
    for level in ['200', '500', '850']:
        print('读%s'%level)
        ds1 = get_one(t, level)
        level_list.append(ds1)
        # print(a)
    ds2 = xr.concat(level_list, pd.Index(pressure_list, name='pressure'))
    return ds2

def main_multi():
    pass
    ttt = pd.date_range(start='2021-07-18 08', end='2021-07-20 20', freq='12H')
    pool = Pool(6)
    result = []
    for t in ttt:
        tr = pool.apply_async(multi_one, args=(t,))
        # print("计算%d"%i)
        result.append(tr)
    pool.close()
    pool.join()

    time_list = []
    for j in result:
        time_list.append(j.get())
    # print(c)
    tttt = ttt-pd.Timedelta('8H')
    ds3 = xr.concat(time_list, pd.Index(tttt, name='time'))
    return ds3

if __name__ == '__main__':
    aa = main_multi()
    aa.to_netcdf('/mnt/zfm_18T/fengxiang/HeNan/Data/OBS/obs_upar_latlon1.nc')
    
    
    
    
