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

# %%
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
import salem
from pyresample import image, geometry
from pyresample.geometry import AreaDefinition,GridDefinition
import h5py
# %%

flnm_save = '/home/fengx20/project/sod/data_combine/'+'tbb_times.nc'
ds = xr.open_dataset(flnm_save)
da = ds['tbb'].isel(time=10)
# da.lat.min()
