#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
对tbb亮温是0.的缺省处理
读卫星的星下点经纬度的处理
np.arange() 对步长为0.1的数据处理产生问题，
使用np.around() 限制小数位
-----------------------------------------
Time             :2021/04/12 08:53:15
Author           :Forxd
Version          :1.0
'''
# %%
import numpy as np
import xarray as xr
from pyresample import image
from pyresample.geometry import GridDefinition
import h5py
import os
import pandas as pd
from multiprocessing import Pool


# %%
def get_lonlat(flnm_hdf, flnm_dat):
    # flnm_hdf = '/home/fengx20/project/sod/data_original/obs/FY2F_20210717_0723/FY2F_TBB_IR1_NOM_20210717_0000.hdf'
    hdfFile = h5py.File(flnm_hdf, 'r')

    ## 读取dat文件中的经纬度信息
    var = list(hdfFile.keys())[0]
    shape = hdfFile[var].shape # 一般为2288*2288
    # flnm_dat = "/home/fengx20/project/sod/data_original/obs/FY2G_20210716_0722/NOM_ITG_2288_2288(0E0N)_LE.dat"  # 该卫星云顶亮温的描述文件
    with open(flnm_dat, 'r') as f:
        data = np.fromfile(f, dtype=np.float32, count=shape[0] * shape[1])  # 读经度lon原数据
        data1 = data.reshape(shape[0], shape[1])  # 根据卫星数据TBB的格点分布,重新分成这个维度
        data = np.fromfile(f, dtype=np.float32, count=shape[0] * shape[1])  # 读纬度lat
        data2 = data.reshape(shape[0], shape[1])

    ## 获得星下点的经纬度, 参考https://blog.csdn.net/weixin_44052055/article/details/116273068
    infoh = hdfFile['/NomFileInfo']
    latCenter = infoh[0][3]
    lonCenter = infoh[0][4]
    ## 根据星下点经度，对原始经纬度数据进行校正,
    # 这里的lat2d周边有的点是没有数据的，也不是规则的
    lon2d = data1 + lonCenter
    lat2d = data2 + latCenter
    return lon2d, lat2d


def resample_satellite(lon2d, lat2d, lon_gridmesh, lat_gridmesh, data):
    """
    lon2d, lat2d [np.array] -> 原始网格经纬度
    lon_gridmesh, lat_gridmesh [np.array] -> 目标网格经纬度
    data [np.array] -> 原始数据
    """
    ### 定义网格点
    ## 原始数据的网格
    grid_def = GridDefinition(lons=lon2d, lats=lat2d)  
    ## 目标网格
    grid_def2 = GridDefinition(lons=lon_gridmesh, lats=lat_gridmesh)  

    ## 利用pyresample进行重采样
    cc = image.ImageContainerNearest(data, grid_def, radius_of_influence=5000) # 变为image类型
    area_con_nn = cc.resample(grid_def2)  # 重采样
    result_data_nn = area_con_nn.image_data
    # xr.DataArray(result_data_nn).plot()

    da = xr.DataArray(
            result_data_nn,
            coords={
                'lon':(('south_north','west_east'),lon_gridmesh),
                'lat':(('south_north','west_east'),lat_gridmesh),
            },
            dims=['south_north', 'west_east']
        )
    return da

def get_time(hdfFile, var):
    """
    处理hdf数据的时间
    不同的卫星数据可能会有变化
    """
    year = str(hdfFile[var][0][7]).zfill(2)
    month = str(hdfFile[var][0][8]).zfill(2)
    day = str(hdfFile[var][0][9]).zfill(2)
    hour = str(hdfFile[var][0][10]).zfill(2)
    str_time = year+'-'+month+'-'+day+' '+hour
    print(str_time)
    t = pd.to_datetime(str_time)
    return t

def get_tbb_obs(flnm_hdf):
    ## 原始网格
    # flnm_hdf = '/home/fengx20/project/sod/data_original/obs/FY2F_20210717_0723/FY2F_TBB_IR1_NOM_20210717_0000.hdf'
    flnm_dat = "/home/fengx20/project/sod/data_original/obs/FY2G_20210716_0722/NOM_ITG_2288_2288(0E0N)_LE.dat"  # 该卫星云顶亮温的描述文件
    lon2d , lat2d = get_lonlat(flnm_hdf, flnm_dat)

    ## 目标网格
    # interval = 0.125
    interval = 0.1
    lon_grid = np.arange(100-1, 120+1+interval, interval)
    lat_grid = np.arange(25-1, 40+1+interval, interval)
    lon_gridmesh, lat_gridmesh = np.meshgrid(lon_grid, lat_grid)

    ## tbb数据
    hdfFile = h5py.File(flnm_hdf, 'r')
    ## 读取dat文件中的经纬度信息
    var = 'FY2F TBB Hourly Product'
    data = hdfFile[var][:,:].round(1)
    da = resample_satellite(lon2d, lat2d, lon_gridmesh, lat_gridmesh,data)

    ## 增加时间维度
    var = list(hdfFile.keys())[1] # 这里是NomFileInfo
    t = get_time(hdfFile, var)
    da.coords['time'] = t
    da = da.rename('tbb')
    return da


def get_tbb_obs_dual(path):
    
    # path = '/home/fengx20/project/sod/data_original/obs/FY2G_20210716_0722/'
    # path = '/home/fengx20/project/sod/data_original/obs/FY2F_20210717_0723/'
    fl_list = os.popen('ls {}/{}*'.format(path, 'FY'))  # 打开一个管道
    fl_list = fl_list.read().split()

    pool = Pool(processes=8)
    results = []
    num = len(fl_list)
    for i in range(num):
        results.append(pool.apply_async(get_tbb_obs, args=(fl_list[i],)))
    pool.close()
    pool.join()


    time_list = []
    for j in results:
        time_list.append(j.get())
    print(time_list[0])
    # tttt = ttt-pd.Timedelta('8H')
    ds = xr.concat(time_list, dim='time')
    
    return ds

if __name__ == "__main__":
    path = '/home/fengx20/project/sod/data_original/obs/FY2F_20210717_0723/'
    ds = get_tbb_obs_dual(path)
    flnm_save = '/home/fengx20/project/sod/data_combine/'+'tbb_times.nc'
    ds.to_netcdf(flnm_save)
    # flnm_hdf = '/home/fengx20/project/sod/data_original/obs/FY2F_20210717_0723/FY2F_TBB_IR1_NOM_20210717_0000.hdf'
    # get_tbb_obs(flnm_hdf)