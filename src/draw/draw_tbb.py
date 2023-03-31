# %%
import xarray as xr
from draw_rain_distribution_24h import get_dr
from baobao.map import get_rgb
import numpy as np

# %%

flnm = '/home/fengx20/project/sod/data_combine/'+'tbb_times.nc'
ds = xr.open_dataset(flnm)
da = ds['tbb'].sel(time='2021-07-20 04')
dr = get_dr()
# dr.
# dr.colorlevel=np.array([-100, -20,-10,0, 5,10, 15, 100])#雨量等级
# dr.colorlevel=np.array([-100, -50,-20,0, 5,10, 15, 100])#雨量等级
# dr.colorlevel=np.array([-200,-70, -65,-60,-55, -50,-40, 100])#雨量等级
dr.colorlevel=np.array([-200,-70, -60,-50, -45,-40,-30, 100])#雨量等级
# fn = '/home/fengx20/project/sod/src/draw/7colors_tbb.txt'
rgbtext = '/home/fengx20/project/sod/src/draw/7colors_tbb.txt'
rgb = get_rgb(rgbtext)
dr.colordict = rgb
dr.colorticks= dr.colorlevel[1:-1]
cf = dr.draw_single(da-273.15)

cb = dr.fig.colorbar(
    cf,
    # cax=ax6,
    orientation='horizontal',
    ticks=dr.colorticks,
    fraction = 0.06,  # 色标大小,相对于原图的大小
    pad=0.1,  #  色标和子图间距离
    )
cb.ax.tick_params(labelsize=10)  # 设置色标标注的大小
labels = list(map(lambda x: str(x) if x<1 else str(int(x)), dr.colorticks))  # 将colorbar的标签变为字符串
cb.set_ticklabels(labels)


# %%