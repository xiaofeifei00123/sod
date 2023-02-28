#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
读取micaps数据的, 地面填图的10m风场，并插值
-----------------------------------------
Time             :2021/12/30 22:35:12
Author           :Forxd
Version          :1.0
'''


import xarray as xr
import wrf
import netCDF4 as nc
import matplotlib.pyplot as plt
# from 
from nmc_met_io.read_micaps import read_micaps_1, read_micaps_2, read_micaps_14
import meteva.base as meb
import numpy as np


import xarray as xr
import meteva.base as meb
import metpy.interpolate as interp
import numpy as np
import pandas as pd
from nmc_met_io.read_micaps import read_micaps_1, read_micaps_2, read_micaps_14
import meteva.base as meb
from nmc_met_graphics.plot import mapview
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xesmf as xe
from multiprocessing import Pool





def read_10wind(dic):
    # flnm = '/home/fengxiang/HeNan/Data/OBS/Micaps/surface/PLOT_10MIN/20210711000000.000'
    # path = '/mnt/zfm_18T/fengxiang/HeNan/Data/OBS/Micaps/surface/PLOT_10MIN/'
    path = '/mnt/zfm_18T/fengxiang/HeNan/Data/OBS/Micaps/surface/WIND_AVERAGE_10MIN_ALL_STATION/'
    flnm = path+str(dic['time'].strftime('%Y%m%d%H%M%S.000'))
    print(flnm)
    df = read_micaps_1(flnm)
    df2 = df.reindex(columns=['ID','time','lon', 'lat', 'wind_angle', 'wind_speed'])
    sta = df2.dropna(axis=0)
    sta['u'] = -1*np.sin(sta['wind_angle']/180*np.pi)*sta['wind_speed']
    sta['v'] = -1*np.cos(sta['wind_angle']/180*np.pi)*sta['wind_speed']
    sta.index.name = 'station'
    sta.columns.name = 'var'
    ds = xr.Dataset(sta)
    ds = ds.swap_dims({'station':'ID'})
    ds = ds.set_coords(['lat', 'lon'])
    t = ds['time'][0]
    ds1 = ds.drop_vars(['time'])
    ds1 = ds1.rename({'ID':'idx'})


    area = {
        'lon1':110.5,
        'lon2':116,
        'lat1':32,
        'lat2':36.5,
        'interval':0.05,
    }
    
    index = ((ds1.lat<=area['lat2']) & (ds1.lat>=area['lat1']) & (ds1.lon>=area['lon1']) & (ds1.lon<=area['lon2']))
    # print(ds1)
    ## 有部分时次某些站点没有数据,没有数据也就是光有站号，没有经纬度，但是作为维度，所有的站点都算了
    # ds2 = ds1.to_array().loc[index]  # 这里没有时间维度
    ds2 = ds1.isel(idx=index)
    # print(ds1)
    # ds2 = ds1.assign_attrs({'time':t.values})
    # print(ds2)
    return ds2




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
        'lon1':110.5,
        'lon2':116,
        'lat1':32,
        'lat2':36.5,
        'interval':0.5,
    }
    ds_regrid = xe.util.grid_2d(area['lon1']-area['interval']/2, 
                                area['lon2'], 
                                area['interval'], 
                                area['lat1']-area['interval']/2, 
                                area['lat2'],
                                area['interval'])

    # ds_regrid = xe.util.grid_2d(area['lon1'], area['lon2'], area['interval'], area['lat1'], area['lat2'], area['interval'])
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


def get_one(t):
    """获得一个时次, 一个层次的插值过后的高空填图数据
    """
    # t = pd.Timestamp('2021-07-19 2000')
    dic_t = {
        'var':'temp',
        # 'level':level,
        'time':t
    }
    ## 读取站点数据
    ds = read_10wind(dic_t)
    # print(ds)
    return ds

    ## 对各个变量分别进行站点插值到格点
    var_list = list(ds.var()) ## 获取Dataset全部变量
    ds_regrid = xr.Dataset()
    for var in var_list:
        da = interp_metpy(ds[var])
        ds_regrid[var] = da
    ds_regrid.assign_attrs({'time':t})
    return ds_regrid
    # return ds

def main():
    # pressure_list = [200, 500, 850]
    ttt = pd.date_range(start='2021-07-19 08', end='2021-07-19 09', freq='1H')
    # ttt = ttt.drop(['2021-07-19 20','2021-07-20 00','2021-07-20 04','2021-07-20 16','2021-07-20 17','2021-07-20 18', '2021-07-20 22', '2021-07-21 07'])
    time_list = []
    for t in ttt:
        # level_list = []
        # for level in ['200', '500', '850']:
            # print('读%s'%level)
        ds1 = get_one(t)
            # level_list.append(ds1)
            # print(a)
        # ds2 = xr.concat(level_list, pd.Index(pressure_list, name='pressure'))
        time_list.append(ds1)
    ds3 = xr.concat(time_list, pd.Index(ttt, name='time'))
    # ds3 = xr.concat(time_list, pd.Index(ttt, name='time'))
    return ds3

def multi_one(t):
    pass
    # pressure_list = [200, 500, 850]
    # level_list = []
    # for level in ['200', '500', '850']:
        # print('读%s'%level)
    ds1 = get_one(t)
        # level_list.append(ds1)
        # print(a)
    # ds2 = xr.concat(level_list, pd.Index(pressure_list, name='pressure'))
    return ds1

def main_multi():
    pass
    ttt = pd.date_range(start='2021-07-19 08', end='2021-07-21 08', freq='1H')
    # ttt = ttt.drop('2021-07-20 18') # 该时次数据有问题
    # ttt = ttt.drop(['2021-07-20 18', '2021-07-20 22', '2021-07-21 07', ])
    # ttt = ttt.drop(['2021-07-19 20','2021-07-20 00','2021-07-20 04','2021-07-20 16','2021-07-20 17','2021-07-20 18', '2021-07-20 22', '2021-07-21 07'])
    ttt = ttt.drop(['2021-07-19 20','2021-07-20 00','2021-07-20 04','2021-07-20 16','2021-07-20 18', '2021-07-20 22', '2021-07-21 07'])
    # ttt = ttt.drop(['2021-07-19 20','2021-07-20 00','2021-07-20 04','2021-07-20 16', '2021-07-20 22', '2021-07-21 07'])
    pool = Pool(12)
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

    ## 将lat,和lon变成一维的, 这一步很重要
    lon = ds3.lon.mean(dim='time').values  # 神来之比
    lat = ds3.lat.mean(dim='time').values
    dda = ds3.drop_vars(['lon', 'lat', 'station'])
    ds3 = dda.assign_coords({'lon':('idx',lon), 'lat':('idx', lat)})
    return ds3

if __name__ == '__main__':
    aa = main_multi()
    # aa = main()
    # print(aa)
    aa.to_netcdf('/mnt/zfm_18T/fengxiang/HeNan/Data/OBS/10m_wind_station.nc')