#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
画多个试验降水在一张图上
难点在于子图的摆放位置等等设置
和上一个的改变，是可以自定义的多了，逻辑简单了
-----------------------------------------
Time             :2021/03/24 11:38:10
Author           :Forxd
Version          :2.0
'''

import xarray as xr
import numpy as np
import salem
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.io.shapereader import Reader, natural_earth
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import geopandas
import cmaps

from read_tbb_obs_dual import get_time_list

# from SAL_domain import get_small_area
# from SAL_V3 import pretreatment
"""将多个试验的结果画在一张图上
"""

def draw_station(ax):
    station = {
        'TR': {
            'lat': 28.6,
            'lon': 87.0
        },
        'NQ': {
            'lat': 31.4,
            'lon': 92.0
        },
        'LS': {
            'lat': 29.6,
            'lon': 91.1
        },
        'TTH': {
            'lat': 34.2,
            'lon': 92.4
        },
        'GZ': {
            'lat': 32.3,
            'lon': 84.0
        },
        'SZ': {
            'lat': 30.9,
            'lon': 88.7
        },
        'SQH': {
            'lat': 32.4,
            'lon': 80.1
        },
        # 'JinChuan': {
        #     'lat': 31.29,
        #     'lon': 102.04
        # },
        # 'JinLong': {
        #     'lat': 29.00,
        #     'lon': 101.50
        # },
    }
    values = station.values()
    station_name = list(station.keys())
    print(type(station_name[0]))
    # print(station_name[0])
    x = []
    y = []
    for i in values:
        y.append(float(i['lat']))
        x.append(float(i['lon']))

    ## 标记出站点
    ax.scatter(x,
               y,
               color='black',
               transform=ccrs.PlateCarree(),
               alpha=1.,
               linewidth=0.2,
               s=10)
    ## 给站点加注释
    for i in range(len(x)):
        print(x[i])
        ax.text(x[i] - 1,
                 y[i] + 0.5,
                 station_name[i],
                 transform=ccrs.PlateCarree(),
                 alpha=1.,
                 fontdict={
                     'size': 9,
                 })

def create_map(ax):
    """创建地图对象
    ax 需要添加底图的画图对象

    Returns:
        ax: 添加完底图信息的坐标子图对象
    """
    proj = ccrs.PlateCarree()
    # --设置地图属性
    # 画省界
    provinces = cfeat.ShapelyFeature(Reader(
        '/mnt/Disk4T_5/fengxiang_file/Data/Map/cn_shp/Province_9/Province_9.shp'
    ).geometries(),
                                     proj,
                                     edgecolor='k',
                                     facecolor='none')

    # 画青藏高原
    Tibet = cfeat.ShapelyFeature(
        Reader('/home/fengxiang/Data/shp_tp/Tibet.shp').geometries(),
        proj,
        edgecolor='k',
        facecolor='none')

    # ax.add_feature(provinces, linewidth=0.6, zorder=2)
    ax.add_feature(Tibet, linewidth=0.6, zorder=2)  # 添加青藏高原区域

    # --设置图像刻度
    ax.set_xticks(np.arange(80, 102 + 2, 4))
    ax.set_yticks(np.arange(26, 38 + 2, 2))
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(axis='both', labelsize=5, direction='out')
    # -- 设置图像范围
    # ax.set_extent([78, 98, 26, 38], crs=ccrs.PlateCarree())
    return ax


def draw_contourf(data, ax, cmap, title):
    # levels = np.arange(0, 66, 5)  # 设置colorbar分层
    levels = [200, 205, 210, 215, 220, 225, 235, 245, 250]  # 需要画出的等值线
    ax = create_map(ax)
    print(title[0])

    x = data.lon
    y = data.lat
    crx = ax.contourf(x,
                      y,
                      data,
                      cmap=cmap,
                      extend='both',
                      levels=levels,
                      transform=ccrs.PlateCarree())
    ax.set_title(title[0], loc='left', y=0.82, fontsize=12)
    ax.set_extent([78, 102, 26, 38], crs=ccrs.PlateCarree())
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    ax.text(99.5, 37, title[1])
    draw_station(ax)
    return crx

def draw_obs(year):

    if str(year) == '2005':
        flnm = "./Data/TBB_2005.nc"
        title = "28 Jul 2005"
        fig_name= "200507.png"
    elif str(year) == '2014':
        flnm = "./Data/TBB_2014.nc"
        title = "19 Aug 2014"
        fig_name= "201408.png"
    else:
        print("输入的时间有误，请检查")
    # title = "Obs_200507_2812_2815"
    # ftime = get_time_list(2005)
    ftime = get_time_list(year)
    ds = xr.open_dataset(flnm)
    # tbb = ds[ftime[0]]
    # Draw(tbb, title, 2005)


    ## 获取地形文件
    shp_file = '/home/fengxiang/Data/shp_tp/Tibet.shp'
    shp = geopandas.read_file(shp_file)

    # tbb[0] = ds[ftime[0]] # 06时
    dic = {}
    dic_title = {}
    title_list = [None]*6
    for i in ftime:
        key = i[-4:-2]
        dic.update({key:i})  # 生成时间的字典
        # str_title = str(key)+"00UTC 28 Jul 2005"
        str_title = str(key)+"00UTC 19 Aug"
        # title_list.append(str_title)
        dic_title.update({key:str_title})  # 生成时间的字典
    print(dic_title)
    

    tbb = [None] * 6
    # time_list = ['06', '11', '12', '13', '14', '23']
    time_list = ['06', '09', '12', '13', '19', '23']
    for i in range(len(time_list)):
        tbb[i] = ds[dic[time_list[i]]]
        title_list[i] = dic_title[time_list[i]]


    ## --->画图
    proj = ccrs.PlateCarree()  # 创建坐标系

    fig = plt.figure(figsize=(8, 7), dpi=400)  # 创建页面
    ## 设置子图位置和大小

    grid = plt.GridSpec(3,
                        2,
                        figure=fig,
                        left=0.07,
                        right=0.96,
                        bottom=0.12,
                        top=0.96,
                        wspace=0.3,
                        hspace=0.3)
    axes = [None] * 6  # 设置一个维度为5的空列表
    # # print(axes)
    axes[0] = fig.add_subplot(grid[0, 0:1], projection=proj)
    axes[1] = fig.add_subplot(grid[0, 1:2], projection=proj)
    axes[2] = fig.add_subplot(grid[1, 0:1], projection=proj)
    axes[3] = fig.add_subplot(grid[1, 1:2], projection=proj)
    axes[4] = fig.add_subplot(grid[2, 0:1], projection=proj)
    axes[5] = fig.add_subplot(grid[2, 1:2], projection=proj)
    # # 设置colorbar位置和名称等
    ccc = cmaps.precip3_16lev_r
    colors = mpl.cm.get_cmap(ccc)
    col = colors(np.linspace(0, 1, 18))
    cccc = mpl.colors.ListedColormap([
        col[0], col[1], col[2],  (231/250, 177/250, 22/250), col[4],
        col[6], '#85f485', '#16c516','white',
    ])

    cmap = cccc
    # print(cmap)

    draw_contourf(tbb[0], axes[0], cmap, [title_list[0], '(a)'])
    draw_contourf(tbb[1], axes[1], cmap, [title_list[1], '(b)'])
    draw_contourf(tbb[2], axes[2], cmap, [title_list[2], '(c)'])
    draw_contourf(tbb[3], axes[3], cmap, [title_list[3], '(d)'])
    draw_contourf(tbb[4], axes[4], cmap, [title_list[4], '(e)'])
    cf = draw_contourf(tbb[5], axes[5], cmap, [title_list[5], '(f)'])

    ax6 = fig.add_axes([0.18, 0.04, 0.7, 0.03])  # 重新生成一个新的坐标图

    # ## 画色标
    # bounds = np.arange(5, 65, 5)
    bounds = [205, 210, 215, 220, 225, 235, 245]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
    # cb = plt.colorbar(cf, orientation='horizontal', ticks=[205, 210, 215, 220, 225, 235, 245],fraction=0.05, pad=    0.1)

    cb = fig.colorbar(
        cf,
        cax=ax6,
        orientation='horizontal',
        ticks=bounds,
        fraction = 0.05,  # 色标大小
        pad=0.1,  # 透明度
        # extend='both'
    )
    
    # fig.suptitle("28 Jul 2005")
    # fig.savefig("200507.png")
    # plt.subplots(constrained_layout=True)
    # plt.tight_layout(pad=0.1, h_pad=0.1)

    # fig.suptitle(title)
    fig.savefig(fig_name)

if __name__ == '__main__':
    draw_obs('2014')
