# %%

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
flnm = '/home/fengx20/project/LeeWave/data/output/cross_GWD3.nc'
ds = xr.open_dataset(flnm)

da = ds['wa_cross']
da1 = da.interpolate_na(dim='vertical', method='linear',  fill_value="extrapolate")
da2 = da1.interp(vertical=[5000], method='linear',  kwargs={"fill_value": "extrapolate"}).squeeze()
# db
# %%
val = da2.values
# val
# da2.dims
# da2
# %%

ft = np.fft.fft2(val)
ft
ft = np.fft.fftshift(ft)
# ft = np.fft.ifftshift(ft)


# x = np.arange(1,ft.shape[1]+1)*30*1000
# %%
val.shape
# y = np.arange(1,ft.shape[0]+1)*3600
# %%
nx = val.shape[1]
ny = val.shape[0]
x = np.linspace(0, 1, nx)  # 将0到1平分成1400份
y = np.linspace(0, 1, ny)  # 将0到1平分成1400份
x = np.arange(nx)  # 将0到1平分成1400份
y = np.arange(ny)  # 将0到1平分成1400份
# y1 = 1/y*ny
# x2 = np.pi*2
# y2 = np.pi*2
# nx
# ny

# %%
# nx
# x1

# %%
# import cmaps
aa = (np.abs(ft)/(nx*ny)*2)**2
# aa = np.abs(ft)/ny*2
bb = np.log(aa)
# bb = aa
# aa.min()
# bb
# aa.min()
# %%
val.max()
# %%
x1 = 1/x*nx
y1 = 1/y*ny
# %%
# x.shape
# nx
# x1
# %%
cm = 1/2.54
fig = plt.figure(figsize=(8*cm, 8*cm), dpi=300)
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
crx2 = ax.contourf(x, y, bb, levels=8)
# crx2 = ax.contourf(x1, y1, bb, levels=10)
# crx2 = ax.contourf(x2, y2, bb, levels=10)

# %%
# x1.min()
# da2
# np.arange(2)
# x1
# x1.max()
# x1
# da2
# bb.shape
# len(x1)
# bb.shape
# len(x1)

