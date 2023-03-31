#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
多次启动的WRF, 即每个试验对应有多个时间段的WRF降水数据处理
单词启动的WRF
读取wrfout数据中的降水
将wrfout数据中的降水集合成一个文件
还可以选择存格点总降水，只要对流降水
# TODO 将wrfout的格点数据插值为站点数据
# TODO 关于接口问题，争取做到只改变文件路径即可
# TODO 用正常人的思路去写代码
# 目前没有进行多进程操作，因为对于每个时次的降水来说，都是要减去前一个时次的，所以多进程相对不好写
速度目前还可以
-----------------------------------------
Time             :2021/09/09 10:53:08
Author           :Forxd
Version          :1.0
'''

# %%
import numpy as np
import pandas as pd
import datetime
import xarray as xr
import os
import xesmf as xe
import numpy as np
import wrf
import netCDF4 as nc
from baobao.interp import regrid_xesmf

# %%
class Rain():

    def combine_rain(self, path, domain='wrfout_d03', flag='all'):
        """将每个时间段的wrfout试验结果合并到一起，算出逐小时降水
        path: 包含wrfout数据的路径
        domain: 合并哪个domain的降水数据呢, wrfout_d01;wrfout_d02;wrfout_d03
        flag: 需要的是， 总降水all、格点降水grid，还是对流降水convection呢

        由于wrfout数据的降水是累计降水，
        这里将它变为逐小时降水,同时进行合并
        又由于wrfout数据的坐标是x,y格点上的，
        通过wrf-python库将其转为不规则的latlon格点坐标

        Args:
            path ([type]): 包含有wrfout数据的文件夹路径

        Returns:
            rain[DataArray] : 多时次聚合后的降水 
        """    

        # fl_list = os.popen('ls {}/wrfout_d03*'.format(path))  # 打开一个管道
        # fl_list = os.popen('ls {}/wrfout_d03*'.format(path))  # 打开一个管道
        fl_list = os.popen('ls {}/{}*'.format(path, domain))  # 打开一个管道
        fl_list = fl_list.read().split()
        dds_list = []
        r = 0
        for fl in fl_list:
            print(fl[-19:])
            ds = xr.open_dataset(fl)
            ## 总降水
            if flag == 'all':
                da = ds['RAINNC']+ds['RAINC']+ds['RAINSH']-r
                r = (ds['RAINNC']+ds['RAINC']+ds['RAINSH']).values.round(1)
            
            ## 对流降水
            elif flag == 'convection':
                da = ds['RAINC']+ds['RAINSH']-r   #  深对流+浅对流
                r = (ds['RAINC']+ds['RAINSH']).values.round(1)
            elif flag == 'grid': 
                # 格点降水
                da = ds['RAINNC']-r   #  
                r = (ds['RAINNC']).values.round(1)
            else:
                print("请输入需要计算的降水类型， all, convection, grid")
                break
            dda = da.squeeze()  # 该是几维的就是几维的
            dc = dda.rename({'XLAT':'lat', 'XLONG':'lon', 'XTIME':'time'})
            dds_list.append(dc)
        da_concate = xr.concat(dds_list, dim='time') # 原始的，未经坐标变化的降水
        rain = da_concate.round(1) 
        return rain

    def grid2station(self, flnm_obs, flnm_wrf, flnm_wrfout):
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

    def regrid_latlon(self, flnm_rain, area):
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
        
    def caculate_area_mean(self, da, area,):
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


class TimesRain(Rain):
    """
    多次冷启动试验的结果，作为一次试验结果的
    """
    def create_fold_times(self, path_main,):
        """每个敏感性试验的文件夹， 创建一个时间文件夹

        Args:
            path_main (_type_): FD试验，SS试验这些的目录
            path_namelist (_type_): _description_
            path_wrf (_type_): _description_
            path_met (_type_): _description_
        """

        tt = pd.date_range('2021-07-16 12', '2021-07-21 12')
        folder_list = []
        fpath_time_list = []
        for t in tt:
            t1 = t
            t2 = t+pd.Timedelta('36H')
            folder_name = t1.strftime('%Y-%m-%d-%H')+'__'+t2.strftime('%Y-%m-%d-%H')
            folder_list.append(folder_name)
            path = os.path.join(path_main, folder_name)  # 每一个具体的试验实施的路径
            # print(path)
            fpath_time_list.append(path)
        return fpath_time_list
        
    def combine_times_1model(self, ):
        """
        对于多次启动试验，
        每个试验单独存为一个文件(其实是有多个冷启动的时间段的结果)
        """
        # path_main = '/mnt/zfm_18T/fengxiang/HeNan/Data/GWD/d03/newall/'
        path_main = '/mnt/zfm_18T/fengxiang/HeNan/Data/GWD/d03/DA/'
        # fd_list = ['CTRL', 'GWD3', 'FD', 'SS']
        fd_list = ['CTRL', 'GWD3']
        for fd in fd_list:
            path = os.path.join(path_main, fd)
            ff = self.create_fold_times(path)

            for path in ff:
                print(path)
                self.save_one_model(path)

    def combine_times_models(self,):
        """
        对于多次启动试验，
        所有试验的降水存为一个文件
        """
        # path_main = '/mnt/zfm_18T/fengxiang/HeNan/Data/GWD/d03/newall/'
        path_main = '/mnt/zfm_18T/fengxiang/HeNan/Data/GWD/d03/DA/'
        flag = 'all'
        # fd_list = ['CTRL',]#  'GWD3', 'FD', 'SS']
        fd_list = ['CTRL','GWD3']
        rain_list = []
        for fd in fd_list:
            path = os.path.join(path_main, fd)
            ff = self.create_fold_times(path)

            rain1_list = []
            for path in ff:
                # print(path)
                flnm= path+'/'+flag+'.nc'
                # ds = xr.open_dataset(flnm)
                ds = xr.open_dataarray(flnm)
                tt = pd.Series(path.split('/')[-1].split('__'))# .astype('np.datetime')#.to_datetime('%y-%m-%d-%h')
                t1 = datetime.datetime.strptime(tt[0], '%Y-%m-%d-%H')+pd.Timedelta('13H')
                t2 = datetime.datetime.strptime(tt[1], '%Y-%m-%d-%H')
                # dss = ds
                dss =ds.sel(time=slice(t1, t2))
                print(t1, t2)
                rain1_list.append(dss)
            rain1 = xr.concat(rain1_list, dim='time')
            rain1 = rain1.rename(fd)
            rain_list.append(rain1)
        ds = xr.merge(rain_list)
        return ds


class TimeRain(Rain):
    """
    普通的，一次启动的wrf数据
    """

    def save_one(self, path_main, domain='wrfout_d03', flag='all'):
        """处理一个模式的数据

        Args:
            path_main (str, optional): [description]. Defaults to '/mnt/zfm_18T/fengxiang/HeNan/Data/1900_90m/'.
        """
        ## 合并数据
        da = self.combine_rain(path_main,domain, flag)
        flnm_save = path_main+'/rain_'+flag+'.nc'
        da.to_netcdf(flnm_save)

        ## 降低分辨率和转换投影
        # da1 = regrid_latlon(path_dic['path_rain_wrf_grid'], area)
        # da1.to_netcdf(path_dic['path_rain_wrf_latlon'])

        ## 插值到站点
        # da2 = grid2station(path_dic['path_rain_obs_station'], path_dic['path_rain_wrf_grid'],path_dic['path_wrfout'])
        # da2.to_netcdf(path_dic['path_rain_wrf_station'])
        # pass

    def store_single(self,):
        pass
        path_main = '/home/fengx20/project/sod/data_original/wrfout/sod_all/'
        da = self.save_one(path_main=path_main)

    def store_combine(self,):
        """处理多个模式的数据
        """
        ##### 对于不同试验需要修改的地方
        model_list = ['sod_all', 'sod_bl', 'sod_fd', 'sod_ls', 'sod_ss', 'sod_no', 'sod_scale_aware', 'sod_ss_fd', 'Run05', 'Run15']
        # model_list = ['sod_scale_aware', 'sod_ss_fd']
        # model_list = ['sod_all','Run05', 'Run15']
        domain = 'wrfout_d03'
        # flag = 'all'
        # flag_list = ['all','convection', 'grid']
        ## 单独保存每个数据
        path = '/home/fengx20/project/sod/data_original/wrfout/'
        path_out = '/home/fengx20/project/sod/data_combine/'
        flag_list = ['all']

        # ###### 以下不需要修改
        print("读取保存单个模式数据")
        for model in model_list:
            print(model)
            path_main = path+model+'/'
            # path_main = '/mnt/zfm_18T/fengxiang/HeNan/Data/GWD/d03/'+model+'/'
            for flag in flag_list:
                self.save_one(path_main, domain, flag)

        ## 将所有数据读取，然后合并保存
        # print("合并多模式数据")
        # for flag in flag_list:
        #     rain_list = []
        #     for model in model_list:
        #         print(model)
        #         path_main = path+model+'/rain_'
        #         # self.save_one(path_main, domain, flag)
        #         flnm = path_main+flag+'.nc'
        #         da = xr.open_dataarray(flnm)
        #         da.name = 'precip'
        #         rain_list.append(da)
        #     ds = xr.concat(rain_list,pd.Index(model_list,name='model'))
        #     ds.to_netcdf(path_out+'/rain_'+flag+'.nc')
            
            

# %%
if __name__ == '__main__':
    tr = TimeRain()
    tr.store_single()
    # tr.store_combine()

    