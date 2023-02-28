#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
站点的高空数据
单站数据的聚合
区域平均值
其实很简单，先插值到一个站点
然后再插值到垂直层
读取单个和多个站的wrf逐层数据
目的是为了多插值几层
不同模式，一个站点的数据插值到一起

分别存储，读取聚合
这样做的好处是可以节省内存，针对每一步的错误分别改进

参考：
https://wrf-python.readthedocs.io/en/latest/user_api/generated/wrf.interp1d.html#wrf.interp1d

## 对于读取的文件可以这样设置不同的纵坐标
dds = ds.set_coords(['height', 'pressure'])
dds.swap_dims({'bottom_top':'pressure'})
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
from baobao.caculate import caculate_q_rh_thetav, caculate_average_wrf

# %%


class Sounding():
    """单个站点, 单个模式的
    诊断量计算
    数据聚合
    """
    def __init__(self,
                model='gwd0', 
                sta_dic={'sta_num':'57083','sta_name':'zhengzhou','lon':113.66,'lat':34.71 },
                path_main = '/mnt/zfm_18T/fengxiang/HeNan/Data/GWD/d03/',
                # path_wrfout ='/mnt/zfm_18T/fengxiang/HeNan/Data/GWD/gwd0/' ,
                # path_save ='/mnt/zfm_18T/fengxiang/HeNan/Data/GWD/gwd0/' ,
                 ):
        """ 存放一些公共使用的变量,
        也就是每个函数都一样的哪些
        将一些不同模式不一样的东西，全部定义到这里
        """
        pass
        self.model = model
        self.sta_dic = sta_dic
        self.path_main = path_main
        # self.path_wrfout = path_wrfout
        # self.path_save = path_save
        self.path_wrfout = os.path.join(path_main, model)
        self.path_save = os.path.join(path_main, model)

    def sounding_1station_1time(self, flnm = '/mnt/zfm_18T/fengxiang/HeNan/Data/1900_90m/wrfout_d03_2021-07-19_00:00:00'):
        # flnm = '/mnt/zfm_18T/fengxiang/HeNan/Data/1900_90m/wrfout_d04_2021-07-19_00:00:00'
        print(flnm[-19:])
        wrfnc = nc.Dataset(flnm)
        # lat, lon = get_centroid()

        # lat = self.sta_dic['lat']
        # lon = self.sta_dic['lon']

        # x,y = wrf.ll_to_xy(wrfnc, 34.71, 113.66)
        # x,y = wrf.ll_to_xy(wrfnc, lat, lon)

        ## south
        # area = {
        #     'lat1':33.2,
        #     'lat2':34,
        #     'lon1':111.8,
        #     'lon2':113.2,
        # }
        ## north
        area = {
            'lat1':34.2,
            'lat2':35.2,
            'lon1':113,
            'lon2':114,
        }
        
        ## 高度， 各层海拔高度, 单位m, 和探空资料保持一致
        hagl = wrf.getvar(wrfnc, 'height', units='m')  # 先纬度后经度
        pj = hagl.attrs['projection'].proj4()
        hagl = hagl.assign_attrs({'projection':pj})
        p = wrf.getvar(wrfnc, 'pres', units='hpa').assign_attrs({'projection':pj})
        u = wrf.getvar(wrfnc, 'ua', units='m/s').assign_attrs({'projection':pj})
        v = wrf.getvar(wrfnc, 'va', units='m/s').assign_attrs({'projection':pj})
        w = wrf.getvar(wrfnc, 'wa', units='m/s').assign_attrs({'projection':pj})
        t = wrf.getvar(wrfnc, 'temp', units='degC').assign_attrs({'projection':pj})
        td = wrf.getvar(wrfnc, 'td', units='degC').assign_attrs({'projection':pj})
        theta = wrf.getvar(wrfnc, 'theta', units='degC').assign_attrs({'projection':pj})
        theta_e = wrf.getvar(wrfnc, 'theta_e', units='degC').assign_attrs({'projection':pj})

        # dd = wrf.getvar(wrfnc, 'DTAUX3D_LS').assign_attrs({'projection':pj})
        # print(dd)
        
        

        d1 = wrf.getvar(wrfnc, 'DTAUX3D_LS').assign_attrs({'projection':pj})
        d2 = wrf.getvar(wrfnc, 'DTAUX3D_SS').assign_attrs({'projection':pj})
        d3 = wrf.getvar(wrfnc, 'DTAUX3D_BL').assign_attrs({'projection':pj})
        d4 = wrf.getvar(wrfnc, 'DTAUX3D_FD').assign_attrs({'projection':pj})
            # dtaux = dtaux.values+dd
        
        # dtaux = xr.DataArray(dtaux,
        #                      coords=t.coords,
        #                      dims=t.dims,
        #                      )
        dtaux = d1+d2+d3+d4
        # print(dtaux)

        
        
        
        


        ## 计算区域平均值
        hagl = caculate_average_wrf(hagl, area=area).round(1)
        p = caculate_average_wrf(p, area=area).round(0)
        u = caculate_average_wrf(u, area=area).round(1)
        v = caculate_average_wrf(v, area=area).round(1)
        w = caculate_average_wrf(w, area=area)
        t = caculate_average_wrf(t, area=area).round(1)
        td = caculate_average_wrf(td, area=area).round(1)
        theta = caculate_average_wrf(theta, area=area).round(1)
        theta_e = caculate_average_wrf(theta_e, area=area).round(1)
        dtaux = caculate_average_wrf(dtaux, area=area)
        

        
        ## 根据u,v风计算风向和风速
        deg = 180.0/np.pi # 角度和弧度之间的转换
        rad = np.pi/180.0

        # wind_speed = xr.ufuncs.sqrt(u**2+v**2).round(1)
        wind_speed = np.sqrt(u**2+v**2).round(1)
        wind_speed.name = 'wind_speed'
        # wind_angle = (180.0+xr.ufuncs.arctan2(u, v)*deg).round(0)
        wind_angle = (180.0+np.arctan2(u, v)*deg).round(0)
        wind_angle.name = 'wind_angle'
        u = u.rename('u')  # 给DataArray一个名称u
        v = v.rename('v')
        w = w.rename('w')
        dtaux = dtaux.rename('dtaux')
        

        # ds = xr.merge([t,td, u, v, p, hagl])
        ds = xr.merge([t,td, u, v, w,wind_speed, wind_angle, p, hagl, theta, theta_e, dtaux])
        dds = ds.set_coords(['height', 'pressure'])
        ds_return = dds.swap_dims({'bottom_top':'pressure'})
        return ds_return

    def sounding_1station(self, fl_list):
        """单进程循环读取文件
        单个站点多个时次
        """
        pass
        dds_list = []
        for fl in fl_list:
            dds = self.sounding_1station_1time(fl,)
        dds_list.append(dds)
        dds_concate = xr.concat(dds_list, dim='Time')
        dds_return = dds_concate.rename({'XLAT':'lat', 'XLONG':'lon', 'Time':'time'}).drop_vars('XTIME')
        return dds_return

    def sounding_1station_mp(self, fl_list):
        """多进程读取文件
        单个站点多个时次
        """
        pass
        pool = Pool(13)
        result = []
        for fl in fl_list:
            tr = pool.apply_async(self.sounding_1station_1time, args=(fl,))
            result.append(tr)
        pool.close()
        pool.join()

        dds_list = []
        for j in result:
            dds_list.append(j.get())
        # print(dds_list)
        dds_concate = xr.concat(dds_list, dim='Time')
        # ds_upar = dds_concate.rename({'level':'pressure', 'XLAT':'lat', 'XLONG':'lon', 'Time':'time'})
        # dds_return = dds_concate.rename({'XLAT':'lat', 'XLONG':'lon', 'Time':'time'}).drop_vars('XTIME')
        dds_return = dds_concate.rename({'Time':'time'}).drop_vars('XTIME')
        return dds_return

    
    def sounding_main(self):
        """处理流程的主控制函数

        Args:
            path ([type]): [description]
            path = '/mnt/zfm_18T/fengxiang/HeNan/Data/ERA5/YSU_1912/'

        Returns:
            [type]: [description]
        """
        # path_wrfout = os.path.join(self.path_main, self.model)
        fl_list = os.popen('ls {}/wrfout_d03*'.format(self.path_wrfout))  # wrfout文件的path
        fl_list = fl_list.read().split()

        print("1.合并不同时次数据")
        dds = self.sounding_1station_mp(fl_list)
        print("2. 开始计算诊断变量")
        cc = caculate_q_rh_thetav(dds)
        print("3. 合并诊断变量")
        ds_upar = xr.merge([dds, cc])
        print("4. 保存单站，单模式，所有时次数据")

        # self.sta_dic={'sta_num':'57083','sta_name':'zhenzhou','lon':113.66,'lat':34.71 }
        # flnm = 'sounding_'+self.sta_dic['sta_name']+'_'+self.model+'.nc'
        flnm = 'time_height'+'_'+self.model+'north.nc'
        path_save = os.path.join(self.path_save,flnm)
        ds_upar.to_netcdf(path_save)
        return ds_upar


# sd = Sounding()
# sd.sounding_main()


# %%

def sounding_dual():
    """
    将wrfout数据中需要的变量聚合成一个文件，并进行相关的垂直插值, 和诊断量的计算
    处理两种模式，不同时次的数据
    多模式数据的合并
    """
    # model_list = ['gwd0', 'gwd1', 'gwd3']
    model_list = ['gwd3']

    # sta_dic_list = [
    #     {'sta_num':'57083','sta_name':'zhengzhou','lon':113.66,'lat':34.71 },
    #     {'sta_num':'57178','sta_name':'nanyang','lon':112.4,'lat':33.1 }, 
    #     {'sta_num':'57067','sta_name':'lushi','lon':111.04,'lat':34.05 }, 
    #                ]
    
    # for stadic in sta_dic_list:
    for model in model_list:
        sd = Sounding(model=model)
        ds = sd.sounding_main()
        # ds_list.append(ds)
        print(ds)

# %%
def combine():
    """将不同模式，不同站点的数据聚合到一起
    """

    model_list = ['gwd3']
    sta_dic_list = [
        {'sta_num':'57083','sta_name':'zhengzhou','lon':113.66,'lat':34.71 },
        {'sta_num':'57178','sta_name':'nanyang','lon':112.4,'lat':33.1 }, 
        {'sta_num':'57067','sta_name':'lushi','lon':111.04,'lat':34.05 }, 
    ]
    sta_list = ['zhengzhou', 'nanyang', 'lushi']
    path_main = '/mnt/zfm_18T/fengxiang/HeNan/Data/GWD/d03/'
    ds_sta_list = []
    for stadic in sta_dic_list:
        ds_model_list = []
        for model in model_list:
            flnm = 'sounding_'+stadic['sta_name']+'_'+model+'.nc'
            # print(path_main)
            path_save1 = os.path.join(path_main, model)
            path_save = os.path.join(path_save1,flnm)
            print(path_save)
            ds = xr.open_dataset(path_save)
            ds_model_list.append(ds)
        dds = xr.concat(ds_model_list, dim=pd.Index(model_list, name='model'))
        ds_sta_list.append(dds)
    ddds  = xr.concat(ds_sta_list, dim=pd.Index(sta_list, name='station'))
    flnm_all = os.path.join(path_main,'sounding_all.nc')
    ddds.to_netcdf(flnm_all)
    return ddds
    # dds


# %%
if __name__ == '__main__':
    pass
    sounding_dual() # 分别存储
    # aa = combine()  # 合并为一个文件
    # print(aa)
    
