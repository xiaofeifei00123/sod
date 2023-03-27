#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
读取多个tbb数据，将它们写入一个nc文件中
-----------------------------------------
Time             :2021/04/12 08:54:44
Author           :Forxd
Version          :1.0
'''

import numpy as np
import xarray as xr
from pandas import Series, DataFrame
import pandas as pd

import cmaps  # 设置色标的
import xesmf as xe  # 插值的
import os  # 操作文件的

from multiprocessing import Pool
from multiprocessing import Manager
import multiprocessing


def read_tbb_bin(flmn):
    """读取卫星的TBB资料，筛选出范围内的数据，返回成DataSet的形式
    Args:
        flmn,观测的TBB的小时数据
    Returns:
        [type]: 返回的是带有坐标属性的DataSet格式的原始卫星TBB数据
    """
    fx = "./NOM_ITG_2288_2288(0E0N)_LE.dat"  # 该卫星云顶亮温的描述文件

    ## ----------------------------------------------
    ## -- 获取卫星资料的描述文件
    ## -- 就是对应的格点数据的经纬度
    ## ----------------------------------------------
    with open(fx, 'r') as f:
        data = np.fromfile(f, dtype=np.float32, count=2288 * 2288)  # 读经度lon原数据
        data1 = data.reshape(2288, 2288)  # 根据卫星数据TBB的格点分布,重新分成这个维度
        data = np.fromfile(f, dtype=np.float32, count=2288 * 2288)  # 读纬度lat
        data2 = data.reshape(2288, 2288)

    # 根据星下点经度，对原始经纬度数据进行校正,
    lonCenter = 104.5  # footpoint of FY-2C\E\F
    latCenter = 0
    # 这里的lat2d周边有的点是没有数据的，也不是规则的
    lon2d = data1 + lonCenter  # 二维得经度坐标
    lat2d = data2 + latCenter

    # print(type(data))
    # print(data1)
    ## ----------------------------------------------
    ## 读TBB数据
    # TBB = xr.open_dataarray(target_dir+target_file)
    TBB = xr.open_dataarray(flmn)
    tbb = TBB.values  # 将云顶亮温转为numpy的DataArray格式

    print(lat2d[500][500])
    print(lat2d[500][501])
    print(lat2d[500][502])
    ## 设置区域范围
    # lat_s = 26
    # lat_n = 38
    # lon_w = 80
    # lon_e = 98

    lat_s = 20
    lat_n = 45
    lon_w = 70
    lon_e = 110
    ## 筛选范围内的云顶亮温, 直接使用numpy切片
    ## 这一步是比较没想到的
    index = np.where((lon2d < lon_e) & (lon2d > lon_w) & (lat2d > lat_s) &
                     (lat2d < lat_n))  # 这个是numpy的where语法，返回的是各维度坐标，组成的元组
    # print(index)
    # tbb[loc] = None
    tbb_reset = tbb[index[0][0]:index[0][-1],
                    index[1][-1]:index[1][0]]  # 筛选的范围内的云顶亮温TBB
    lat_reset = lat2d[index[0][0]:index[0][-1],
                      index[1][-1]:index[1][0]]  # 筛选的范围内的云顶亮温TBB
    lon_reset = lon2d[index[0][0]:index[0][-1],
                      index[1][-1]:index[1][0]]  # 筛选的范围内的云顶亮温TBB
    # print(lon_reset)
    # print(lat_reset)
    # 将数组变成连续的，加快读写速度
    tbb_reset = np.ascontiguousarray(tbb_reset)
    lat_reset = np.ascontiguousarray(lat_reset)
    lon_reset = np.ascontiguousarray(lon_reset)

    # loc = np.where(tbb_reset<100)
    # tbb_reset[loc] = None
    # print(tbb_reset[loc])
    # 为TBB赋坐标属性
    tbb_return = xr.Dataset(
        {'tbb': (['south_north', 'west_east'], tbb_reset)},
        coords={
            'lat': (['south_north', 'west_east'], lat_reset),
            'lon': (['south_north', 'west_east'], lon_reset),
        },
        attrs={'variable': 'tbb'},
    )

    # print(tbb_return)
    print(tbb_return['tbb'])
    return tbb_return


def regrid(dataset):
    """利用xESMF库，将非标准格点的数据，插值到标准格点上去
    Args:
        dataset ([type]): Dataset格式的数据, 这里是由TBB构成的DataSet
    """

    ## 创建ds_out, 利用函数创建,这个东西相当于掩膜一样
    # ds_out = xe.util.grid_2d(79.875,98,0.25,25.875,38,0.25) # 80-98E,26-38N
    # ds_out = xe.util.grid_2d(79.95,98,0.1,25.95,38,0.1) # 80-98E,26-38N
    # ds_out = xe.util.grid_2d(79.95, 102, 0.1, 25.95, 38, 0.1)  # 80-98E,26-38N
    ds_out = xe.util.grid_2d(77.95, 105, 0.1, 25.95, 38, 0.1)  # 80-98E,26-38N
    regridder = xe.Regridder(dataset, ds_out, 'bilinear')  # 好像是创建了一个掩膜一样
    dp = dataset['tbb']  # 获取变量
    dp_out = regridder(dp)  # 返回插值后的变量

    dp_val = dp_out.values
    loc = np.where(dp_val < 100)
    dp_val[loc] = None

    ## 将二维的附属变量(经纬度), 变成一维的, 为了画图方便
    # np.around(data,n) 保留n位小数
    # lat = np.around(np.arange(26, 38 + 0.01, 0.1), 1)
    # lon = np.around(np.arange(80, 102 + 0.01, 0.1), 1)
    lat = np.around(np.arange(26, 38 + 0.01, 0.1), 1)
    lon = np.around(np.arange(78, 105 + 0.01, 0.1), 1)
    # print(lon[-1])
    # dp_return_dataarray = xr.DataArray(dp_out.values, coords=[lat, lon], dims=('lat', 'lon'))
    dp_return_dataarray = xr.DataArray(dp_val,
                                       coords=[lat, lon],
                                       dims=('lat', 'lon'))
    # # print(dp_return)
    # dp_return_dataset = xr.Dataset({"tbb": dp_return_dataarray})
    # # return dp_return_dataset
    return dp_return_dataarray


def get_tbb_obs(fnm):
    """得到插值过后的tbb

    Args:
        fnm ([type]): tbb原始文件

    Returns:
        dp_dataset: 插值后的tbb
    """

    tbb_origin = read_tbb_bin(fnm)  # 从原始数据中读出来的

    tbb_regrid = regrid(tbb_origin)  # 插值到标准经纬度上

    # dp_dataset = xr.Dataset({"tbb": tbb_regrid})
    # return dp_dataset
    return tbb_regrid
def get_time_list(year):
    file_time_2014 = [
        '0819_0600',
        '0819_0700',
        '0819_0800',
        '0819_0900',
        '0819_1000',
        '0819_1100',
        '0819_1200',
        '0819_1300',
        '0819_1400',
        '0819_1500',
        '0819_1900',
        '0819_2000',
        '0819_2100',
        '0819_2200',
        '0819_2300',
        '0820_0000',
    ]
    file_time_2005 = [
        '0728_0600',
        '0728_0700',
        '0728_0900',
        '0728_1000',
        '0728_1100',
        '0728_1200',
        '0728_1300',
        '0728_1400',
        '0728_1500',
        '0728_1600',
        '0728_1700',
        '0728_1800',
        '0728_1900',
        '0728_2000',
        '0728_2100',
        '0728_2200',
        '0728_2300',
        '0729_0000',
        '0729_0100',
        '0729_0200',
        '0729_0300',
    ]
    if str(year) == '2014':
        return file_time_2014
    if str(year) == '2005':
        return file_time_2005


def get_dataset(year):
    ## 2014年8月19日09-15时TBB平均值
    target_dir = "/mnt/zfm/Fengx/Assess_pbl_data/TBB_data/"+str(year)+"/"
    # target_file = "FY2C_TBB_IR1_NOM_20050729_0300.hdf"  # 卫星云顶亮温
    if str(year) == '2005':
        str_file = "FY2C_TBB_IR1_NOM_2005"
    elif str(year) == '2014':
        str_file = "FY2E_TBB_IR1_NOM_2014"
    else:
        print("输入日期有误")

    # file_time = ['0819_1000',]
    file_time = get_time_list(year)
    num = len(file_time)
    pool = Pool(processes=num)
    results = []


    flnm = []
    for i in file_time:
        fnm = target_dir+str_file+i+".hdf"
        flnm.append(fnm)

    for i in range(num):
        results.append(pool.apply_async(get_tbb_obs, args=(flnm[i],)))
    pool.close()
    pool.join()

    dic = {}
    for i in range(num):
        ddr = results[i].get()
        print(ddr)
        dic.update({file_time[i]:ddr})
    ds = xr.Dataset(dic)
    # print(ds)
    ds.to_netcdf("./Data/TBB_"+str(year)+".nc")



if __name__ == '__main__':

    # get_dataset('2014')
    get_dataset('2005')