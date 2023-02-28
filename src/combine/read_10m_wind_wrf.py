#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
读取wrfout数据中的降水
将wrfout数据中的降水集合成一个文件
# TODO 将wrfout的格点数据插值为站点数据
-----------------------------------------
Time             :2021/09/09 10:53:08
Author           :Forxd
Version          :1.0
'''

# %%
import xarray as xr
import os
import xesmf as xe
import numpy as np
import wrf
import netCDF4 as nc
from baobao.interp import regrid_xesmf
import netCDF4 as nc
# %%

# %%
def combine_rain(path):
    """
    由于wrfout数据的降水是累计降水，
    这里将它变为逐小时降水,同时进行合并
    又由于wrfout数据的坐标是x,y格点上的，
    通过wrf-python 库将其转为不规则的latlon格点坐标

    Args:
        path ([type]): 包含有wrfout数据的文件夹路径

    Returns:
        rain[DataArray] : 多时次聚合后的降水 
    """    

    fl_list = os.popen('ls {}/wrfout_d03*'.format(path))  # 打开一个管道
    fl_list = fl_list.read().split()
    dds_list = []
    r = 0
    for fl in fl_list:
        print(fl[-18:])

        # flnm = '/mnt/zfm_18T/fengxiang/HeNan/Data/GWD/d03/gwd0/wrfout_d01_2021-07-19_12:00:00'
        wrfnc = nc.Dataset(fl)
        uv = wrf.getvar(wrfnc, 'uvmet10')
        uv1 = uv.rename({'XLAT':'lat', 'XLONG':'lon', 'XTIME':'time', })
        pj = uv1.attrs['projection'].proj4()
        dc = uv1.assign_attrs({'projection':pj})
        
        

        # ds = xr.open_dataset(fl)
        # da = ds['RAINNC']+ds['RAINC']+ds['RAINSH']-r
        # r = (ds['RAINNC']+ds['RAINC']+ds['RAINSH']).values.round(1)
        # wrfnc = nc.Dataset(fl)
        # uv = wrf.getvar(wrfnc, 'uvmet10')
        # u = uv.sel(u_v='u')

        

        # p = wrf.getvar(wrfnc, 'pres', units='hpa')[:,x,y].assign_attrs({'projection':pj}).drop_vars(['latlon_coord'])
        # dc = dda.rename({'XLAT':'lat', 'XLONG':'lon', 'XTIME':'time'})
        # u = wrf.getvar(wrfnc, 'ua', units='m/s')[:,x,y].assign_attrs({'projection':pj}).drop_vars(['latlon_coord'])

        # dda = da.squeeze()  # 该是几维的就是几维的
        # dc = dda.rename({'XLAT':'lat', 'XLONG':'lon', 'XTIME':'time'})
        dds_list.append(dc)
    da_concate = xr.concat(dds_list, dim='time') # 原始的，未经坐标变化的降水
    rain = da_concate.round(1)
    return rain

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
    ds_obs = xr.open_dataset(flnm_obs)
    da_obs = ds_obs['u']
    # ds_obs = xr.open_dataset(flnm_obs)
    # da_obs = ds_obs['u']
    # wrf输出后聚合到一起的多时间维度数据
    da_wrf = xr.open_dataarray(flnm_wrf)

    cc = da_obs.isel(time=0)
    # sta = cc.id.values
    wrfin = nc.Dataset(flnm_wrfout)
    x, y = wrf.ll_to_xy(wrfin, cc.lat, cc.lon)

    # x = xr.where(x>-1, x, np.nan)
    # y = xr.where(y>-1, y, np.nan)
    # # print(y)
    # x = x.dropna(dim='idx').astype(np.int16)
    # y = y.dropna(dim='idx').astype(np.int16)
    # # print(y)

    # print(x[0])
    # print(x,y)
    da_station = da_wrf.sel(south_north=y, west_east=x)
    da_station = da_station.assign_coords({'sta':('idx', da_obs.idx.values)})
    da_station = da_station.swap_dims({'idx':'sta'})
    # print(da_station)
    da_station = da_station.drop_vars(['latlon_coord'])
    # da_station = da_station.rename({'idx':'sta'})
    # print(da_station)

    return da_station


# %%
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
    ds_out = regrid_xesmf(ds, area, rd=1)
    return ds_out
    

def save_one(path_main = '/mnt/zfm_18T/fengxiang/HeNan/Data/1900_90m/'):
    """处理一个模式的数据

    Args:
        path_main (str, optional): [description]. Defaults to '/mnt/zfm_18T/fengxiang/HeNan/Data/1900_90m/'.
    """

    path_dic = {
        'path_main':path_main,  # 模式数据文件夹
        'path_wrfout':'/mnt/zfm_18T/fengxiang/HeNan/Data/GWD/d03/gwd3-BL/wrfout_d03_2021-07-19_19:00:00', # 原始的一个wrfout数据，获得投影需要
        'path_rain_wrf_grid':path_main+'10m_wind.nc', # 原始降水数据存储路径+文件名
        'path_rain_wrf_latlon':path_main+'10m_wind_latlon.nc',  # 插值到latlon之后的文件名
        'path_rain_wrf_station':path_main+'10m_wind_station.nc',  # 插值到站点之后的文件名
        'path_rain_obs_station':'/mnt/zfm_18T/fengxiang/HeNan/Data/OBS/10m_wind_station.nc', # 站点降水
    }
    area = {
        'lon1':110.5,
        'lon2':116,
        'lat1':32,
        'lat2':36.5,
        'interval':0.05,
    }

    ## 合并数据
    da = combine_rain(path_main)
    da.to_netcdf(path_dic['path_rain_wrf_grid'])

    ## 降低分辨率和转换投影
    # da1 = regrid_latlon(path_dic['path_rain_wrf_grid'], area)
    # da1.to_netcdf(path_dic['path_rain_wrf_latlon'])

    ## 插值到站点
    da2 = grid2station(path_dic['path_rain_obs_station'], path_dic['path_rain_wrf_grid'],path_dic['path_wrfout'])
    da2.to_netcdf(path_dic['path_rain_wrf_station'])
    pass

def dual():
    """处理多个模式的数据
    """
    pass
    # model_list = ['gwd0', 'gwd1', 'gwd3','gwd3-FD', 'gwd3-BL','gwd3-SS', 'gwd3-LS']
    # model_list = ['gwd0', 'gwd1', 'gwd3']
    model_list = ['weak_typhoon', 'strengthen_typhoon']
    for model in model_list:
        # path_main = '/mnt/zfm_18T/fengxiang/HeNan/Data/GWD/d03/'+model+'/'
        path_main = '/mnt/zfm_18T/fengxiang/HeNan/Data/Typhoon/'+model+'/'
        # print(path_main)
        # path_main = '/mnt/zfm_18T/fengxiang/HeNan/Data/GWD/d04/'+model+'/'
        save_one(path_main)
    
if __name__ == '__main__':

    pass
    # main()
    # save_one()
    dual()




