#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
画弗劳德数的水平分布
-----------------------------------------
Time             :2023/03/07 10:05:31
Author           :Forxd
Version          :1.0
'''

# %%
from draw_rain_distribution_24h import Draw
import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from baobao.get_cmap import get_rgb
# %%
flnm = '/home/fengx20/project/sod/data_caculate/fr.nc'
ds = xr.open_dataset(flnm)
ds
# %%
da = ds['fr'].sel(model='sod_all').sel(bottom_top=2).isel(Time=6)
da = da.rename({'XLAT':'lat', 'XLONG':'lon'})
# dr = Draw()
cm = 1/2.54
proj = ccrs.PlateCarree()  # 创建坐标系
fig = plt.figure(figsize=(8*cm, 8*cm), dpi=300)
ax = fig.add_axes([0.13,0.1,0.82,0.8], projection=proj)
dr = Draw(fig, ax)



dr.colorlevel=[0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 100]
rgbtxt = '/home/fengx20/project/sod/src/draw/white_green_red_9colors.rgb'
rgb = get_rgb(rgbtxt)
dr.colordict = rgb
dr.colorticks=dr.colorlevel[1:-1]


cf = dr.draw_single(da)    
cb = dr.fig.colorbar(
    cf,
    # cax=ax6,
    orientation='horizontal',
    ticks=dr.colorticks,
    fraction = 0.06,  # 色标大小,相对于原图的大小
    pad=0.1,  #  色标和子图间距离
    )
cb.ax.tick_params(labelsize=10)  # 设置色标标注的大小
# labels = list(map(lambda x: str(x) if x<1 else str(int(x)), dr.colorticks))  # 将colorbar的标签变为字符串
labels = list(map(lambda x: str(x) if x<1 else str(x), dr.colorticks))  # 将colorbar的标签变为字符串
cb.set_ticklabels(labels)
# return dr
