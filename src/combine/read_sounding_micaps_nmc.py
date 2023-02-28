#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
单个站的高空数据
获取想要的站点的探空多时次的探空数据集合
57083 113.66 34.71 112 864
   994.4   11.2   24.4   24.4   54   2
   978.8   25.9   23.6   23.6   78   5.1
   978.8   25.8   23.6   23.6   78   5.1
   948.1   55.2   22.1   22.1   73   16.2
   931.1   68.9   21.2   21.2   88   15.2
   735.6   268.7   11.3   11.3   135   17.1
   气压  位势高度(十米)  温度   露点    风向    风速
保留需要的站点的原始数据
一个站不同时次的
合并多个时次和多个站点的数据为一个文件

注意：
    这个是把一个站点不同时次的所有数据都聚合到一块了，
    所以会有很多nan值, 只需要使用这个命令读取即可：
    单个站点为Dataarray, 多个站点为Dataset
    da = xr.open_dataarray('/mnt/zfm_18T/fengxiang/HeNan/Data/OBS/micaps_sounding_station.nc')
    da = da.isel(time=0).dropna(dim='pressure')
-----------------------------------------
Time             :2021/11/07 14:06:22
Author           :Forxd
Version          :1.1
'''

# %%
import xarray as xr
import numpy as np
import os,sys
import pandas as pd
# from baobao.caculate import caculate_q_rh_thetav
from baobao.caculate import caculate_q_rh_thetaev
from nmc_met_io.read_micaps import read_micaps_5


# %%
# flnm = '/mnt/zfm_18T/fengxiang/HeNan/Data/OBS/Micaps/high/TLOGP/20210720140000.000'


# def read_1sta_1time(flnm, id=57083):
#     """单个时次单个站点的DataArry

#     Args:
#         flnm ([type]): [description]
#     """
#     df = read_micaps_5(flnm)
#     ddf = df.loc[df.ID==str(id)]
#     df1 = ddf.loc[:,['pressure', 'height', 'temp', 'td', 'wind_angle', 'wind_speed']]
#     df1 = df1.astype('float')
#     df1.columns.name = 'vars'  # 还必须是这个名称
#     df2 = df1.drop_duplicates('pressure', keep='last')
#     df2 = df2.set_index('pressure')
#     da = xr.DataArray(df2)
#     da = da.assign_coords({'time':df.time[0]})
#     da = da.assign_coords({'id':id})
#     dda = da.dropna(dim='pressure', how='any')
#     return dda


# %%
class GetMicaps():
    
    def __init__(self, ):
        pass
        # self.station = station
        # self.station_number = station['number']
        # self.path_micaps = '/mnt/zfm_18T/fengxiang/HeNan/Data/OBS/Micaps/high/TLOGP/'

    def read_1sta_1time(self, flnm, id=57083):
        """单个时次单个站点的DataArry

        Args:
            flnm ([type]): [description]
        """
        df = read_micaps_5(flnm)
        ddf = df.loc[df.ID==str(id)]
        ## 这里是利用read_micaps_5, 只能是这些名称
        df1 = ddf.loc[:,['pressure', 'height', 'temperature', 'dewpoint', 'wind_angle', 'wind_speed']]
        df1 = df1.astype('float')

        df1['u'] = -1*np.sin(ddf['wind_angle']/180*np.pi)*ddf['wind_speed']
        df1['v'] = -1*np.cos(ddf['wind_angle']/180*np.pi)*ddf['wind_speed']
        df1['height'] = df1['height']*10
        df1.rename(columns={'temperature':'temp', 'dewpoint':'td'}, inplace=True)


        df1.columns.name = 'vars'  # 还必须是这个名称, vars不能变
        df2 = df1.drop_duplicates('pressure', keep='last')
        df2 = df2.set_index('pressure')
        da = xr.DataArray(df2)
        da = da.assign_coords({'time':df.time[0]})
        # da = da.assign_coords({'time':ddf.time.values[0]})
        # da = da.assign_coords({'lon':ddf.lon.values[0]})
        # da = da.assign_coords({'lat':ddf.lat.values[0]})
        da = da.assign_coords({'id':id})
        dda = da.dropna(dim='pressure', how='any').round(1)

        ## 更改变量名，计算u,v风
        # daa = dda.rename({'temperature':'temp', 'dewpoint':'td'})
        # print(dda)
        # daa = dda.drop_sel(vars=['wind_speed', 'wind_angle'])

        return dda

    def read_1st(self, path_micaps, id=57083):
        """读一个站点,多时次的探空数据

        Args:
            id (int, optional): [description]. Defaults to 57083.

        Returns:
            [type]: [description]
        """
        
        fl_list = os.popen('ls {}202107*.000'.format(path_micaps))  # 打开一个管道
        fl_list = fl_list.read().split()
        aa = fl_list
        aa.sort()  # 排一下顺序，这个是对列表本身进行操作
        da_time = []  # 每个时次变量的列表
        ttt = []  # 时间序列列表

        for flnm in aa:
            fl_time = flnm[-18:-8]
            tt = pd.to_datetime(fl_time, format='%Y%m%d%H')
            ## 转为世界时
            tt = tt - pd.Timedelta(hours=8)
            print(tt)

            file_name = os.path.join(path_micaps, flnm)
            da = self.read_1sta_1time(file_name, id=id)
            if len(da)>0:
                da_time.append(da)  
                ttt.append(tt)
            # da_time.append(da)  
        da_return = xr.concat(da_time, pd.Index(ttt, name='time'))
        return da_return


flnm =  '/mnt/zfm_18T/fengxiang/HeNan/Data/OBS/Micaps/high/TLOGP/20210711080000.000'
df = read_micaps_5(flnm)
df.time.values[0]
# %%
# df.time[0]
id = 57083
ddf = df.loc[df.ID==str(id)]
ddf
# %%
ddf.time
# # ddf.time.values[0]
# gm = GetMicaps()
# gm.read_1sta_1time(flnm)
# %%

def one_station(sta='57083'):

    path_micaps = '/mnt/zfm_18T/fengxiang/HeNan/Data/OBS/Micaps/high/TLOGP/'
    gm = GetMicaps()
    # da = gm.read_1st(path_micaps, 57083)
    da = gm.read_1st(path_micaps, sta)
    save_name = '/mnt/zfm_18T/fengxiang/HeNan/Data/OBS/micaps_sounding_station_%s.nc'%sta
    da.to_netcdf(save_name)
    return da

fpath = '/home/fengxiang/aa/2014_07_11-12/data/radar/read_radar/1_Z9396_ZhuMaDian/0/2014*'    
fdir = os.popen('ls -d {}'.format(fpath))
list(fdir)
# %%
    
    


def dual_station():
    pass
    # sta_list = ['zhengzhou', 'nanyang']   # 郑州和南阳
    sta_dic_list = [
        {'sta_num':'57083','sta_name':'zhengzhou','lon':113.66,'lat':34.71 },
        {'sta_num':'57178','sta_name':'nanyang','lon':112.4,'lat':33.1 }, 
        {'sta_num':'57067','sta_name':'lushi','lon':111.04,'lat':34.05 }, 
        {'sta_num':'57447','sta_name':'EnShi','lon':113.66,'lat':34.71 },
        {'sta_num':'57461','sta_name':'YiChang','lon':112.4,'lat':33.1 }, 
        {'sta_num':'57494','sta_name':'WuHan','lon':111.04,'lat':34.05 }, 
    ]
    sta_list = ['zhengzhou', 'nanyang', 'lushi', 'EnShi', 'YiChang', 'WuHan']
    # ds = xr.Dataset()
    ds_list = []
    for sta in sta_dic_list:
        da = one_station(sta['sta_num'])
        print(da)
        ds1 = da.to_dataset(dim='vars')

        cc = caculate_q_rh_thetaev(ds1)
        ds2 = xr.merge([ds1, cc])
        # da = xr.merge([ds1, cc]).to_array(dim='vars')
        # print(da)
        ds_list.append(ds2)

        # ds[sta['sta_name']] = da
    ddds  = xr.concat(ds_list, dim=pd.Index(sta_list, name='station'))
    ## 添加高度坐标为维度属性
    ddds = ddds.set_coords(['height', 'pressure'])
    # print(ds)
    # cc = caculate_q_rh_thetav(ds)
    # ds_upar = xr.merge([ds, cc])
    save_name = '/mnt/zfm_18T/fengxiang/HeNan/Data/OBS/micaps_sounding_station_all.nc'
    ddds.to_netcdf(save_name)

if __name__ == '__main__':
    pass
    # one_station()
    dual_station()


    # path_micaps = '/mnt/zfm_18T/fengxiang/HeNan/Data/OBS/Micaps/high/TLOGP/'
    # gm = GetMicaps()
    # da = gm.read_1st(path_micaps, 57083)
    # print(da)
    # da.to_netcdf('/mnt/zfm_18T/fengxiang/HeNan/Data/OBS/micaps_sounding_station.nc')
# da
