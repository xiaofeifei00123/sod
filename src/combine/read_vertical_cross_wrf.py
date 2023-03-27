#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
读取wrfout的垂直剖面图的数据
-----------------------------------------
Time             :2021/11/09 20:51:01
Author           :Forxd
Version          :1.0
'''
# %%
import xarray as xr
import numpy as np
import pandas as pd
from wrf import getvar, CoordPair, vertcross, get_cartopy
import wrf
from netCDF4 import Dataset
from multiprocessing import Pool
# import matplotlib.pyplot as plt
# from matplotlib.cm import get_cmap
# import cartopy.crs as crs
# from cartopy.feature import NaturalEarthFeature
from baobao.caculate import caculate_pdiv3d, caculate_pvor3d, caculate_div3d, caculate_vor3d
import sys
import os
sys.path.append('../caculate')
from common import Common

# %%
# flnm = '/mnt/zfm_18T/fengxiang/HeNan/Data/GWD/d03/newall/GWD3/wrfout/cross4_1time.nc'
# ds = xr.open_dataset(flnm)
# ds
# %%
# ds.attrs['cross_start']
# ds['drag_cross']
# ds.cross_start
# # a=CoordPair(lat=33, lon=111.5)
# # type(a)
# # type(a.latlon_str())
# # %%
# # ds.to_netcdf('test.nc')
# wrf_file = '/mnt/zfm_18T/fengxiang/HeNan/Data/GWD/d03/gwd3/wrfout_d01_2021-07-19_17:00:00'
# ncfile = Dataset(wrf_file)
# # ncfile
# aa = getvar(ncfile, 'ua')
# bb = aa.attrs['projection']
# %%
# bb.cartopy()

# bb.proj4()
# %%
class CrossData():
    """获得垂直方向切向的数据
    提供剖面数据
    地形的剖面数据
    """
    def __init__(self, wrf_file):
        pass

        ## 伏牛山东西侧
        # self.cross_start = CoordPair(lat=33, lon=111.5)
        # self.cross_end = CoordPair(lat=36, lon=114.3)
        com = Common()
        self.cross_start = CoordPair(lat=com.cross_start[1], lon=com.cross_start[0])
        self.cross_end = CoordPair(lat=com.cross_end[1], lon=com.cross_end[0])

        # self.cross_start = CoordPair(lat=31, lon=112.4)
        # self.cross_end = CoordPair(lat=34, lon=112.4)
        # self.cross_start= CoordPair(lat=33.5, lon=114.5)
        # self.cross_end= CoordPair(lat=35.5, lon=112.5)

        # self.cross_start = CoordPair(lat=32, lon=111.2)
        # self.cross_end = CoordPair(lat=36.5, lon=114.8)
        ## 降水的分布
        # self.cross_start = CoordPair(lat=32, lon=111.5)
        # self.cross_end = CoordPair(lat=36.5, lon=114.5)
        # 伏牛山
        # self.cross_start = CoordPair(lat=34, lon=110.5)
        # self.cross_end = CoordPair(lat=33.5, lon=113)


        # 嵩山
        # self.cross_start = CoordPair(lat=34.5, lon=112.5)
        # self.cross_end = CoordPair(lat=34, lon=115)



        # self.cross_start = CoordPair(lat=36, lon=111)
        # self.cross_end = CoordPair(lat=32.5, lon=113)

        # self.cross_start = CoordPair(lat=35.5, lon=113)
        # self.cross_end = CoordPair(lat=33.5, lon=113.5)
        ## read the ncfile
        # wrf_file = '/mnt/zfm_18T/fengxiang/HeNan/Data/1900_90m/wrfout_d03_2021-07-20_08:00:00'
        self.ncfile = Dataset(wrf_file)
        ## 计算垂直坐标, 可以是离地高度、气压等
        # self.vert = getvar(self.ncfile, "height_agl")  # 离地高度坐标
        self.vert = getvar(self.ncfile, "z")  # 离地高度坐标
        # self.vert = getvar(self.ncfile, "pres")/100  # 气压坐标

    def get_vcross(self, var):
        """获得单个变量的切向数据, 竖着切

        Args:
            var ([type]): 变量名, 需要是wrf-python支持的

        Returns:
            [type]: [description]
        """

        # var =  getvar(self.ncfile, var)
        var_vcross = vertcross(var, self.vert, wrfin=self.ncfile,
                                     start_point=self.cross_start,
                                        end_point=self.cross_end, 
                                        latlon=True, )
        ## 改变投影的attrs的格式
        pj = var_vcross.attrs['projection'].proj4()
        var_vcross = var_vcross.assign_attrs({'projection':pj})


        ## 改变xy_loc的coords的存储格式
        coord_pairs = var_vcross.coords["xy_loc"].values
        x_labels = [pair.latlon_str(fmt="{:.3f}, {:.3f}")
                    for pair in coord_pairs]
        var_vcross = var_vcross.assign_coords({'xy_loc':('cross_line_idx',x_labels)})
        return var_vcross

    def get_ter(self,):
        """获得地形高度
        """
        ter = wrf.getvar(self.ncfile, "ter", timeidx=-1)
        ter_line = wrf.interpline(ter, wrfin=self.ncfile, 
                            start_point=self.cross_start,
                            end_point=self.cross_end)
        ter_line = ter_line.assign_attrs({'projection':'lambert'})
        return ter_line

    def get_cross_data(self, var_list=['ua', 'va', 'wa', 'theta_e', 'theta']):
        """获得垂直切一刀的数据

        Returns:
            [type]: [description]
        """
        da_cross_list = []
        for var in var_list:
            var =  getvar(self.ncfile, var)  # 直接传变量, 不要传文件
            da = self.get_vcross(var)
            da_cross_list.append(da)

        ## 增加计算散度, 三维的散度计算
        u =  getvar(self.ncfile, 'ua')
        v =  getvar(self.ncfile, 'va')
        lon = u.XLONG
        lat = u.XLAT
        div = caculate_pdiv3d(u,v, lon, lat) # 计算得到div
        vor = caculate_pvor3d(u,v, lon, lat) # 计算得到div
        ws = np.sqrt(u**2+v**2).rename('ws')


        # vor1 =
        # vor = caculate_vor3d(u,v, lon, lat) # 计算得到div
        # div = caculate_div3d(u,v, lon, lat) # 计算得到div
        # print((div.values-div1.values).max())
        # print((vor.values-vor1.values).max())

        # dragx0 = getvar(self.ncfile, 'DTAUX3D_LS')
        # dragy0 = getvar(self.ncfile, 'DTAUY3D_LS')
        # dragx1 = getvar(self.ncfile, 'DTAUX3D_SS')
        # dragy1 = getvar(self.ncfile, 'DTAUY3D_SS')
        # dragx2 = getvar(self.ncfile, 'DTAUX3D_BL')
        # dragy2 = getvar(self.ncfile, 'DTAUY3D_BL')
        # dragx3 = getvar(self.ncfile, 'DTAUX3D_FD')
        # dragy3 = getvar(self.ncfile, 'DTAUY3D_FD')
        # drag0 = np.sqrt(dragx0**2+dragy0**2)
        # drag1 = np.sqrt(dragx1**2+dragy1**2)
        # drag2 = np.sqrt(dragx2**2+dragy2**2)
        # drag3 = np.sqrt(dragx3**2+dragy3**2)
        # drag = (drag1+drag2+drag3+drag0).rename('drag')

        

        ws.attrs = u.attrs  # 因为get_vcross,对project做了统一处理，这里需要是一样的
        div.attrs = u.attrs  # 因为get_vcross,对project做了统一处理，这里需要是一样的
        vor.attrs = u.attrs  # 因为get_vcross,对project做了统一处理，这里需要是一样的
        # drag.attrs = u.attrs  # 因为get_vcross,对project做了统一处理，这里需要是一样的
        div_cross = self.get_vcross(div)  # 插值到剖面上
        vor_cross = self.get_vcross(vor)  # 插值到剖面上
        # drag_cross = self.get_vcross(drag)
        ws_cross = self.get_vcross(ws)
        da_cross_list.append(div_cross)
        da_cross_list.append(vor_cross)
        # da_cross_list.append(drag_cross)
        da_cross_list.append(ws_cross)
        ds = xr.merge(da_cross_list)
        ds.attrs['cross_start'] = self.cross_start.latlon_str()
        ds.attrs['cross_end'] = self.cross_end.latlon_str()
        return ds

    def get_proj(self):
        z = wrf.getvar(self.ncfile, "z")
        pj = get_cartopy(z)
        return pj

# %%
# def save_one_model():
#     path = '/mnt/zfm_18T/fengxiang/HeNan/Data/1900_90m/'
#     # tt = pd.date_range('2021-07-20 0000', '2021-07-20 1200', freq='3H')
#     # tt = pd.date_range('2021-07-20 0000', '2021-07-20 1200', freq='12H')
#     tt = pd.date_range('2021-07-20 0000', '2021-07-20 0000', freq='12H')
#     # tt
#     fl_list = []
#     for t in tt:
#         # fl = 'wrfout_d03_'+t.strftime('%Y-%m-%d_%H:%M:%S')
#         fl = 'wrfout_d02_'+t.strftime('%Y-%m-%d_%H:%M:%S')
#         flnm = path+fl
#         fl_list.append(flnm)

#     ds_list = []
#     for fl in fl_list:
#         print(fl[-19:])
#         cd = CrossData(fl)
#         var_list = []
#         ds = cd.get_cross_data()
#         ds_list.append(ds)
#     ds = xr.concat(ds_list, dim='Time')    

#     ## 再把地形高度读出来, 利用了上面的fl和cd, 所以位置不能变
#     ter = cd.get_ter()
#     ds['ter'] = ter

#     ds = ds.rename({'Time':'time'})
#     save_name = path+'cross1.nc'
#     ds.to_netcdf(save_name)


# %%
def __cross_1model_1time(flnm):
    """子函数，多进程中的每一个进程处理的过程
    """
    print(flnm[-19:])
    cd = CrossData(flnm)
    ds = cd.get_cross_data()
    return ds

def save_one_model_mp(path):

    # path = '/mnt/zfm_18T/fengxiang/HeNan/Data/1900_90m/'
    # tt = pd.date_range('2021-07-20 0000', '2021-07-20 0000', freq='1H')
    # tt = pd.date_range('2021-07-20 1200', '2021-07-20 1200', freq='1H')
    tt = pd.date_range('2021-07-20 0100', '2021-07-20 0100', freq='1H')
    # tt = pd.date_range('2021-07-17 0000', '2021-07-23 0000', freq='1H')
    # tt
    # fl_list = []
    # for t in tt:
    #     fl = 'wrfout_d03_'+t.strftime('%Y-%m-%d_%H:%M:%S')
    #     flnm = path+fl
    #     fl_list.append(flnm)

    fl_list = os.popen('ls {}/wrfout_d03*'.format(path))  # 打开一个管道
    fl_list = fl_list.read().split()[0:5]

    pool = Pool(13)
    result = []
    for fl in fl_list:
        tr = pool.apply_async(__cross_1model_1time, args=(fl,))
        result.append(tr)
    pool.close()
    pool.join()

    dds_list = []
    for j in result:
        dds_list.append(j.get())
    ds = xr.concat(dds_list, dim='Time')


    ## 将地形高度加到数据里面去
    fl = fl_list[0]
    cd = CrossData(fl)
    ter = cd.get_ter()
    ds['ter'] = ter

    ds = ds.rename({'Time':'time'})
    # save_name = path+'cross4_times.nc'
    save_name = path+'cross_1time.nc'
    # save_name = path+'cross5_d03_1time.nc'
    ds.to_netcdf(save_name)

def save_all_model():
    model_list = ['sod_all', 'sod_bl', 'sod_fd', 'sod_ss', 'sod_ss', 'sod_no']
    path_main = '/home/fengx20/project/sod/data_original/wrfout/'
    for model in model_list:
        print(model)
        path = path_main+model+'/'
        save_one_model_mp(path)


if __name__ == '__main__':
    pass
    save_all_model()
