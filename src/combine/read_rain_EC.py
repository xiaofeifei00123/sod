#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
读取EC细网格数据
不需要插值，它本身就是标准格点数据
将wrfout和micaps降水插值到它上面
-----------------------------------------
Time             :2021/09/09 11:41:05
Author           :Forxd
Version          :1.0
'''
# %%
import xarray as xr
import meteva.base as meb
import numpy as np
import os
import pandas as pd

# %%
def get_rain_ec():
    path = '/mnt/zfm_18T/fengxiang/HeNan/Data/OBS/EC_thin_grid'
    ## 19号20时起报的， 每隔3h一次
    fl_list = os.popen('ls {}/21071920*'.format(path))  # 打开一个管道
    fl_list = fl_list.read().split()
    dds_list = []
    for fl in fl_list:
        print(fl)
        grd = meb.read_griddata_from_micaps4(fl)

        ## 获得世界时
        str_time_delta = str(grd.dtime[0].values)+'H'
        # print(grd.time)
        # print(str_time_delta)
        tt = grd.time+pd.Timedelta(str_time_delta)-pd.Timedelta('8H')  # 数据是北京时的，化为世界时
        print(tt)
        
        ## 获得精简过后的数据
        aa = grd.squeeze()
        da = xr.DataArray(
            aa.values,
            coords={
                'lat':aa.lat.values,
                'lon':aa.lon.values,
                'time':tt[0].values
            },
            dims=['lat', 'lon'],
        )
        
        dds_list.append(da)
    dds_concate = xr.concat(dds_list, dim='time')
    dds_concate
    return dds_concate



# %%
# ### 测试结束
def save_rian_ec():
    da = get_rain_ec()
    da.to_netcdf('/mnt/zfm_18T/fengxiang/HeNan/Data/OBS/rain_ec.nc')



# %%
if __name__ == '__main__':
    pass
    save_rian_ec()
# %%