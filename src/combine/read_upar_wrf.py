#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
多个点的高空数据，500hPa高度场
读取聚合wrf高空数据
保存
原始网格数据
插值成latlon网格数据
combine:
    从原始的wrfout数据中，
    读取需要的变量
    插值到需要的气压坐标上
    计算需要的诊断变量，例如q
    聚合成一个文件
regrid:
    将combine的数据，水平插值到需要的等经纬网格点上
    或者说是转换为latlon坐标
此程序读取和最终生成的wrfout变量有
u, v, q, temp, height_agl, geopt
-----------------------------------------
Time             :2021/10/05 22:53:08
Author           :fengxiang
Version          :1.1
'''


# %%
import xarray as xr
import os
import xesmf as xe
import numpy as np
import pandas as pd
import netCDF4 as nc
import wrf
from multiprocessing import Pool
# from read_global import caculate_diagnostic, regrid_xesmf
from baobao.caculate import caculate_q_rh_thetav# , caculate_vo_div_wrf
from baobao.interp import regrid_xesmf
# from baobao.coord_transform import xy_ll

# from wrf import getvar

# %%
class GetUpar():
    """获得wrfout高空数据，原始投影
    """
    def get_upar_one(self, fl):
        pre_level = [900, 925, 850, 700, 600, 500, 400,300,200]
        dds = xr.Dataset()
        data_nc = nc.Dataset(fl)
        print(fl[-19:])
        p = wrf.getvar(data_nc, 'pressure', squeeze=False)

        for var in ['ua', 'va', 'wa','td', 'temp', 'theta','theta_e', 'height_agl', 'z','geopt']:
            if var in ['temp', 'td', 'theta', 'theta_e']:
                da = wrf.getvar(data_nc, var, squeeze=False, units='degC')
            else:
                da = wrf.getvar(data_nc, var, squeeze=False)
            # dds[var] = da.expand_dims(dim='Time')
            dds[var] = wrf.interplevel(da, p, pre_level, squeeze=False)
            ## 试图添加投影
            attr_proj = str(dds[var].projection)
            dds[var]=dds[var].assign_attrs({'projection':attr_proj})
        # dds['height_agl']=dds['height_agl'].assign_attrs({'projection':'lambert'})
        return dds

    def get_upar_dual(self, fl_list):
        """单进程循环读取文件
        """
        pass
        dds_list = []
        for fl in fl_list:
            # dds = xr.Dataset()
            dds = self.get_upar_one(fl)
        dds_list.append(dds)
        dds_concate = xr.concat(dds_list, dim='Time')
        dds_return = dds_concate.rename({'level':'pressure', 'XLAT':'lat', 'XLONG':'lon', 'Time':'time'})
        return dds_return

    def get_upar_multi(self, fl_list):
        """多进程读取文件
        """
        pass
        pool = Pool(13)
        result = []
        for fl in fl_list:
            tr = pool.apply_async(self.get_upar_one, args=(fl,))
            result.append(tr)
        pool.close()
        pool.join()

        dds_list = []
        for j in result:
            dds_list.append(j.get())

        dds_concate = xr.concat(dds_list, dim='Time')
        ds_upar = dds_concate.rename({'level':'pressure', 'XLAT':'lat', 'XLONG':'lon', 'Time':'time'})
        # ds_upar = dds_concate.rename({'level':'pressure', 'Time':'time'})
        ds_upar = ds_upar.drop_vars(['XTIME'])
        return ds_upar

        # flnm = '/mnt/zfm_18T/fengxiang/HeNan/Data/1900_90m/wrfout_d04_2021-07-19_01:00:00'
        # latlon = xy_ll(fl)  # 获得所有点的经纬度坐标

        # ds2 = ds_upar.assign_coords({'lat':('south_north',latlon['lat']), 'lon':('west_east',latlon['lon'])})
        # ds3 = ds2.swap_dims({'south_north':'lat', 'west_east':'lon'})
        # ds_return = ds_upar.rename({})
        # return ds_return


    def get_upar(self, path):
        pass
        # path = '/mnt/zfm_18T/fengxiang/HeNan/Data/ERA5/YSU_1912/'
        fl_list = os.popen('ls {}/wrfout_d03*'.format(path))  # 打开一个管道
        # fl_list = os.popen('ls {}/wrfout/wrfout_d03*'.format(path))  # 打开一个管道
        fl_list = fl_list.read().split()
        ## 临时测试
        # fl_list = fl_list[0:5]
        dds = self.get_upar_multi(fl_list)
        print("开始计算诊断变量")
        # dd = caculate_diagnostic(dds)
        cc = caculate_q_rh_thetav(dds)
        print("合并保存数据")
        ds_upar = xr.merge([dds, cc])
        # ds_upar = xr.merge([ds_upar, dd])
        print(ds_upar)
        return ds_upar

def regrid_one(model='1900_90m'):
    """
    将combine得到的数据，插值到latlon格点上
    将二维的latlon坐标水平插值到一维的latlon坐标上
    """
    area = {
        'lon1':110.5,
        'lon2':116,
        'lat1':32,
        'lat2':36.5,
        'interval':0.05,
    }
    path_main = '/mnt/zfm_18T/fengxiang/HeNan/Data/GWD/d03/'
    flnm = 'upar1.nc'
    path_in = path_main+model+'/'+flnm
    ds = xr.open_dataset(path_in)
    # ds_out = regrid_xesmf(ds, area)
    ds_out = regrid_xesmf(ds, area)  # 保留一位小数
    path_out = path_main+model+'/'+'upar_latlon.nc'
    # ds_out = ds_out.rename({'ua':'u', 'va':'v', 'geopt':'height'})
    ds_out = ds_out.rename({'ua':'u', 'va':'v', 'wa':'w'})
    ds_out.to_netcdf(path_out)

def regrid_dual():
    pass
    # model_list = ['1900_90m','1900_900m', '1912_90m', '1912_900m']
    # model_list = ['gwd0', 'gwd1', 'gwd3']
    model_list = ['gwd0', 'gwd3']
    for model in model_list:
        regrid_one(model)

def combine_one(path_main, model='1912_90m'):
    """
    将wrfout数据中需要的变量聚合成一个文件，并进行相关的垂直插值, 和诊断量的计算
    处理两种模式，不同时次的数据
    """
    # path_main = '/mnt/zfm_18T/fengxiang/HeNan/Data/Typhoon/'
    # path_main = '/mnt/zfm_18T/fengxiang/HeNan/Data/GWD/d03/newall/'
    # path_main = '/mnt/zfm_18T/fengxiang/HeNan/Data/GWD/d03/DA/'
    # path_main = os.path.join(path_main, model)
    gu = GetUpar()
    # path_wrfout = path_main+'1912_90m'
    path_wrfout = path_main+model
    print(path_wrfout)
    ds = gu.get_upar(path_wrfout)
    # print("***"*10)
    # print(ds)
    # ds = gu.get_upar_multi(path_wrfout)
    # flnm = model+'upar.nc'
    # path_save = path_main+flnm
    path_save = os.path.join(path_main, model, 'upar.nc')
    # path_save = os.path.join(path_save, )
    # print(path_save)
    ds.to_netcdf(path_save)
    # return ds

def combine():
    """
    将wrfout数据中需要的变量聚合成一个文件，并进行相关的垂直插值, 和诊断量的计算
    处理两种模式，不同时次的数据
    """
    path_main = '/home/fengx20/project/sod/data_original/wrfout/'
    path_save = '/home/fengx20/project/sod/data_combine/'
    model_list = ['sod_all', 'sod_bl', 'sod_fd', 'sod_ss', 'sod_ss', 'sod_no']
    print("单独保存每个模式的数据")
    for model in model_list:
        print(model)
        combine_one(path_main, model)

    print("合并多个模式的数据")
    ds_list = []
    for model in model_list:
        path_save = os.path.join(path_main, model, 'upar.nc')
        ds = xr.open_dataset(path_save)
        ds_list.append(ds)
    ds2 = xr.concat(ds_list,pd.Index(model_list,name='model'))
    ds2.to_netcdf(path_save+'/upar.nc')
        # ds
        # ds.to_netcdf(path_save+'upar.nc')



# %%
if __name__ == '__main__':
    ### combine和regrid一般不同时进行
    combine()
    # regrid_dual()
    # combine_one()
    # regrid_one()
    # combine() 
    # regrid()
