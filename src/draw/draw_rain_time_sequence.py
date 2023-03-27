# %%
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('/home/fengx20/mypy/baobao/my.mplstyle')
# %%
flnm = '/home/fengx20/project/sod/data_caculate/rain_time_sequence.nc'
ds = xr.open_dataset(flnm)
# %%
# ds = ds.sel(time=slice('2021-07-20 00', '2021-07-21 00'))
ds = ds.sel(time=slice('2021-07-19 18', '2021-07-21 00'))
ds
# ds

# %%
rain_model = ds['sod_all']
rain_obs = ds['OBS']
#%%
# rain_model.time.values = (rain_model.time+pd.Timedelta('4H')).values
# xlabels = rain_model.time.dt.strftime('%m%d %H')
# %%
x = np.arange(rain_model.shape[0])
xticks = rain_model.time+pd.Timedelta('8H')
xlabels = xticks.dt.strftime('%d/%H').values.astype('str')
x
# xlabels
# ticks = np.arange()
# %%
# xticks.values
x = np.arange(len(xticks))
# xlabels[0]
# x
# %%
cm = 1/2.54
fig = plt.figure(figsize=(8*cm, 6*cm), dpi=300)
ax = fig.add_axes([0.18, 0.3, 0.75, 0.6])
ax.plot(x, rain_model, label='CTRL', color='green')
# ax.plot(x+5, rain_model, label='CTRL+5', color='green', linestyle='-')
ax.plot(x, rain_obs, label='OBS', color='black' )
ax.set_xticks(x[::5])
ax.set_xticklabels(xlabels[::5], rotation=30)
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
ax.legend(edgecolor='white', fontsize=10)

ax.set_ylabel('Preciptation (mm)')
ax.set_xlabel('Time (Date/Hour)')
figpath = '/home/fengx20/project/sod/picture/picture_rain/'
fig.savefig(figpath+'tiem_sequence')




# %%
