#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
修改地形数据文件
-----------------------------------------
Time             :2023/02/16 10:21:03
Author           :Forxd
Version          :1.0
'''

# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('~/mypy/baobao/my.mplstyle')

from draw_terrain import get_hgt_met, draw_contourf_latlon
from common import Common
import wrf
import netCDF4 as nc
# %%

# %%
def select_area_height(fl_input, multiple):
    """筛选区域范围内的高度数据
    可以直接返回hgtt画图用
    """
    wrfin = nc.Dataset(fl_input)
    com = Common()
    ds1 = xr.open_dataset(fl_input)

    x1, y1 = wrf.ll_to_xy(wrfin, com.area_terrain['lat1'], com.area_terrain['lon1'])
    x2, y2 = wrf.ll_to_xy(wrfin, com.area_terrain['lat2'], com.area_terrain['lon2'])
    hgt_select = ds1['HGT_M'][:,y1.values:y2.values+1,x1.values:x2.values+1]


    if multiple >= 1.0:
        print('地形高度变化的系数是%s'%multiple)
        hgt_modify = hgt_select*multiple
        ds1['HGT_M'][:,y1.values:y2.values+1,x1.values:x2.values+1] = hgt_modify
    elif multiple < 1.0 and multiple>0.0:
        print('地形高度变化的系数是%s'%multiple)
        hgt_modify = hgt_select*multiple
        hgt_modify= xr.where(hgt_modify<200, 200, hgt_modify)
        ds1['HGT_M'][:,y1.values:y2.values+1,x1.values:x2.values+1] = hgt_modify
    elif multiple == 0.0:
        print('地形高度变化的系数是%s'%multiple)
        # hgt_modify= xr.where(hgt_modify>0, 200, hgt_modify)
        # print('0.000000')
        hgt_modify= 200
        ds1['HGT_M'][:,y1.values:y2.values+1,x1.values:x2.values+1] = hgt_modify


    # lat = ds1['XLAT_M'][:,y1.values:y2.values,x1.values:x2.values].squeeze()
    # lon = ds1['XLONG_M'][:,y1.values:y2.values,x1.values:x2.values].squeeze()
    # hgt_m = hgt_select.squeeze()
    # hgtt = hgt_m.assign_coords({'lat':(['south_north', 'west_east'],lat.values),
    #                     'lon':(['south_north', 'west_east'],lon.values)})
    # # draw_contourf_latlon(hgtt, {'title':'origin'})  # 可以直接用来画小图的数据
    return ds1['HGT_M'], hgt_modify, hgt_select 
    # return ds1['HGT_M']

def move_area_height(fl_input, multiple):
    """筛选区域范围内的高度数据
    可以直接返回hgtt画图用
    """
    wrfin = nc.Dataset(fl_input)
    com = Common()
    ds1 = xr.open_dataset(fl_input)

    x1, y1 = wrf.ll_to_xy(wrfin, com.area_terrain['lat1'], com.area_terrain['lon1'])
    x2, y2 = wrf.ll_to_xy(wrfin, com.area_terrain['lat2'], com.area_terrain['lon2'])
    hgt_select = ds1['HGT_M'][:,y1.values:y2.values+1,x1.values:x2.values+1]


    if multiple >= 1.0:
        print('地形高度变化的系数是%s'%multiple)
        hgt_modify = hgt_select*multiple
        ds1['HGT_M'][:,y1.values:y2.values+1,x1.values:x2.values+1] = hgt_modify
    elif multiple < 1.0 and multiple>0.0:
        print('地形高度变化的系数是%s'%multiple)
        hgt_modify = hgt_select*multiple
        hgt_modify= xr.where(hgt_modify<200, 200, hgt_modify)
        ds1['HGT_M'][:,y1.values:y2.values+1,x1.values:x2.values+1] = hgt_modify
    elif multiple == 0.0:
        print('地形高度变化的系数是%s'%multiple)
        # hgt_modify= xr.where(hgt_modify>0, 200, hgt_modify)
        # print('0.000000')
        hgt_modify= 200
        ds1['HGT_M'][:,y1.values:y2.values+1,x1.values:x2.values+1] = hgt_modify


    # lat = ds1['XLAT_M'][:,y1.values:y2.values,x1.values:x2.values].squeeze()
    # lon = ds1['XLONG_M'][:,y1.values:y2.values,x1.values:x2.values].squeeze()
    # hgt_m = hgt_select.squeeze()
    # hgtt = hgt_m.assign_coords({'lat':(['south_north', 'west_east'],lat.values),
    #                     'lon':(['south_north', 'west_east'],lon.values)})
    # # draw_contourf_latlon(hgtt, {'title':'origin'})  # 可以直接用来画小图的数据
    return ds1['HGT_M'], hgt_modify, hgt_select 
    # return ds1['HGT_M']

if __name__ == "__main__":
    # %%
    flpath = '/home/fengx20/project/LeeWave/data/temporary/'
    fl_input = flpath+'/geo_em.d03.nc'
    fl_output = flpath+'/get_em_modify.nc'

    ds = xr.open_dataset(fl_input)
    hgt0 = get_hgt_met(fl_input)
    ds_hgt, hgt_modify, hgt_select = select_area_height(fl_input, 0.5)  
    # ds_hgt, hgt_modify, hgt_select = select_area_height(fl_input, 0.0)  


    ## 修改原始数据，保存成新的文件
    ds['HGT_M'].values = ds_hgt.values
    ds.to_netcdf(fl_output)

    hgt2 = get_hgt_met(fl_output)
    draw_contourf_latlon(hgt2, {'title':'output'})
    # %%







