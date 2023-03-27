"""
画等值线图的
读取标准经纬度线的降水、云顶亮温这种数据,然后画contourf图
并且做到去除青藏高原地形外的数据，画出青藏高原廓线
其实这个程序干的事情还是蛮多的哦
"""

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
import geopandas
import cmaps


def create_map():
    """用来画地图的, 利用cartopy的库

    Returns:
        [type]: 返回的是坐标图的对象，直接用就可以画图了
    """
    # --创建画图空间
    proj = ccrs.PlateCarree()  # 创建坐标系
    fig = plt.figure(figsize=(8, 6), dpi=400)  # 创建页面
    ax = fig.subplots(1, 1, subplot_kw={'projection': proj})
    # ax = fig.add_subplot()

    Tibet = cfeat.ShapelyFeature(
        Reader('/home/fengxiang/Data/shp_tp/Tibet.shp').geometries(),
        proj,
        edgecolor='k',
        facecolor='none')
    ax.add_feature(Tibet, linewidth=0.6, zorder=2)
    # gl.xlabels_top = gl.ylabels_right = gl.ylabels_left = gl.ylabels_bottom = False  # 关闭经纬度标签
    # --设置刻度
    ax.set_xticks(np.arange(80, 98 + 2, 2))
    ax.set_yticks(np.arange(26, 38 + 2, 2))
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.tick_params(axis='both', labelsize=5, direction='out')
    # -- 设置范围
    # ax.set_extent([80, 98, 26, 38], crs=ccrs.PlateCarree())
    ax.set_extent([82, 102, 26, 38], crs=ccrs.PlateCarree())
    return ax


def Draw(flnm, title):
    ## 读取变量
    # ds = xr.open_dataset("./Data/totrain_station.nc")
    # ds = xr.open_dataset("./Data/TBB_2005_072815_2903.nc")
    # ds = xr.open_dataset("./Data/TBB_2005_0728_12_15.nc")
    ds = xr.open_dataset(flnm)
    rain_total = ds.tbb
    print(rain_total)

    ## 获取被地形文件mask后的数据
    shp_file = '/home/fengxiang/Data/shp_tp/Tibet.shp'
    shp = geopandas.read_file(shp_file)
    rain2 = rain_total.salem.roi(shape=shp)  # 利用salem库对数据进行mask
    # rain2 = rain_total

    ## 开始画图, 生成图像对象
    proj = ccrs.PlateCarree()  # 设置地图投影方式
    fig = plt.figure(figsize=(6, 8), dpi=400)  # 创建页面

    ## 创建色标
    ccc = cmaps.precip3_16lev_r
    colors = mpl.cm.get_cmap(ccc)
    col = colors(np.linspace(0, 1, 18))
    cccc = mpl.colors.ListedColormap([
        col[0], col[1], col[2],  (231/250, 177/250, 22/250), col[4],
        col[6], col[8], 'white'
    ])
    # 画填色图, 注意225直接到了245
    levels = [200, 205, 210, 215, 220, 225, 235, 245, 250]  # 需要画出的等值线
    cf = rain2.plot.contourf(ax=create_map(),
                             cmap=cccc,
                             levels=levels,
                             transform=ccrs.PlateCarree(),
                             extend='both')

    ## 画色标colorbar
    ax = plt.colorbar(cf, orientation='horizontal', ticks=[205, 210, 215, 220, 225, 235, 245],fraction=0.05, pad=0.1)
    # ax = plt.colorbar(cf, orientation='horizontal',fraction=0.05, pad=0.1)

    ## 调整图片大小
    plt.subplots_adjust(top=0.90, bottom=0.08, right=0.99, left=0.02)
    # plt.title("200507-28_12_15")
    # plt.savefig("200507-28_12_15.png")
    plt.title(title)
    plt.savefig('./Picture/' + title + ".png")


if __name__ == '__main__':

    flnm = "./Data/TBB_Obs_2812_15.nc"
    title = "Obs_200507_2812_2815"
    Draw(flnm, title)

    # flnm1 = "./Data/TBB_Obs_2812.nc"
    # title1 = "Obs_200507_2812"

    # flnm2 = "./Data/TBB_Obs_2816_2903.nc"
    # title2 = "Obs_200507_2815_2903"

    # Draw(flnm1, title1)