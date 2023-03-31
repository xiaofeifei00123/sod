#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
计算理查森数和Brunt-vaisala频率
考虑饱和及未饱和两种情况下的计算，最后合并结果
参考（Du and Zhang, 2019）
-----------------------------------------
Time             :2023/03/28 16:13:50
Author           :Forxd
Version          :1.0
'''


# %%
import os
import xarray as xr
import pandas as pd
import numpy as np
import wrf
import netCDF4 as nc
from metpy.calc import brunt_vaisala_frequency_squared,second_derivative,first_derivative
# from metpy.calc import *
import metpy.calc as cal
from metpy.units import units
import matplotlib.pyplot as plt
from baobao.timedur import timeit
from baobao.caculate import caculate_profile_wind
import metpy.constants as ms
import atmos

import sys
sys.path.append('../combine')
from read_vertical_cross_wrf import CrossData

sys.path.append('../draw')
from draw_vertical_cross import DrawVertical





# %%
def me2xr(var):
    # rh = cal.relative_humidity_from_specific_humidity(pressure,temperature,q)
    flnm = '../../data_original/wrfout/sod_all/wrfout_d03_2021-07-20_00:20:00'
    wrfin = nc.Dataset(flnm)
    ds = xr.Dataset()
    ds['u'] = wrf.getvar(wrfin, 'ua')
    lon = ds.XLONG.values
    lat = ds.XLAT.values
    # if var.magnitude:
    if str(type(var)) == "<class 'pint.quantity.build_quantity_class.<locals>.Quantity'>":
        var = var.magnitude

    if len(var.shape) == 3:
        da_rh =  xr.DataArray(
                var,
                coords={
                    'XLONG':(('south_north', 'west_east'),lon),
                    'XLAT':(('south_north', 'west_east'),lat),
                    # 'bottom_top':(('bottom_top','south_north', 'west_east'),lat),

                },
                dims =['bottom_top','south_north', 'west_east']
            )
    else:
        da_rh = var
    return da_rh

# def caculate_average_wrf(da, area = {'lat1':33, 'lat2':34, 'lon1':112, 'lon2':113,}):
def caculate_average_wrf(da, area = {'lat1':32, 'lat2':35, 'lon1':111, 'lon2':114,}):
    """求wrfout数据，在区域area内的区域平均值

    Args:
        da ([type]): 直接利用wrf-python 从wrfout中得到的某个变量的数据

        area = {
            'lat1':33,
            'lat2':34,
            'lon1':111.5,
            'lon2':113,
        }
    Returns:
        [type]: [description]
    """
    lon = da['XLONG'].values
    lat = da['XLAT'].values
    # lon = da['lon'].values
    # lat = da['lat'].values
    ## 构建掩膜, 范围内的是1， 不在范围的是nan值
    clon = xr.where((lon<area['lon2']) & (lon>area['lon1']), 1, np.nan)
    clat = xr.where((lat<area['lat2']) & (lat>area['lat1']), 1, np.nan)
    da = da*clon*clat
    da_mean = da.mean(dim=['south_north', 'west_east'])
    return da_mean


def caculate_br_saturated(ds, ):
    ### 公式中用到的常量
    ## 以下全部数组处理成np.array
    g = ms.earth_gravity.magnitude
    L = ms.water_heat_vaporization.magnitude  # 潜热, L
    Rv = ms.water_gas_constant.magnitude*10**3  # 气体常数, Rv, joule/(kelvin kilogram)
    R = ms.dry_air_gas_constant.magnitude# 气体常数, Rv
    cp = ms.dry_air_spec_heat_press.magnitude  # Specific heat at constant pressure for dry air(joule/(kelvin kilogram))

    ### atmos计算的温度单位是K, 气压单位Pa
    T = ds['temp'].values+273.15
    P = ds['pressure'].values*100
    z = ds['z'].values

    es = atmos.equations.es_from_T_Bolton(T)
    qs = atmos.equations.rvs_from_p_es(P, es)
    rv = qs
    rt = atmos.equations.rt_from_rv(rv)
    qw = rt
    xi = R/Rv  # \xi

    ## 计算饱和布朗特维萨拉频率
    term1 = 1+L*qs/R*T
    term2 = 1+xi*(L**2)*qs/(cp*R*T**2)
    # TODO 这个theta使用metpy还是atmos算？, 两个相差0.几度，可以不用考虑
    theta = atmos.equations.theta_from_p_T(P, T)
    term31 = (1/theta*cal.first_derivative(theta, axis=0, x=z))
    term32 = (cal.first_derivative(qs,axis=0,x=z))
    term3 = term31*L/cp/T*term32
    term4 = cal.first_derivative(qw,axis=0,x=z)
    Nm2 = (g*((term1/term2)*term3-term4)).magnitude

    da_Nm2 = me2xr(Nm2)
    return da_Nm2

def caculate_br_unsaturated(ds):
    pressure = units.Quantity(ds.pressure.values, "hPa")
    dew_point = units.Quantity(ds['td'].values, "degC")
    temperature = units.Quantity(ds['temp'].values, "degC")

    q = cal.specific_humidity_from_dewpoint(pressure, dew_point)
    w = cal.mixing_ratio_from_specific_humidity(q)
    theta_v = cal.virtual_potential_temperature(pressure, temperature, w)
    z = ds['z']
    Nm = 9.86/theta_v*cal.first_derivative(theta_v, axis=0, x=z)
    da_Nmus = me2xr(Nm)
    return da_Nmus

def caculate_rh(ds):
    pressure = units.Quantity(ds.pressure.values, "hPa")
    dew_point = units.Quantity(ds['td'].values, "degC")
    temperature = units.Quantity(ds['temp'].values, "degC")
    q = cal.specific_humidity_from_dewpoint(pressure, dew_point)
    w = cal.mixing_ratio_from_specific_humidity(q)
    rh = cal.relative_humidity_from_specific_humidity(pressure,temperature,q)
    da_rh = me2xr(rh)
    return da_rh

def caculate_ri(Nm2, ds):

    ## 计算理查森数
    u = ds['u'].values
    v = ds['v'].values
    z = ds['z'].values

    flnm = '../../data_original/wrfout/sod_all/wrfout_d03_2021-07-20_00:20:00'
    cd = CrossData(flnm)
    lat1 = cd.cross_start.lat
    lat2 = cd.cross_end.lat
    lon1 = cd.cross_start.lon
    lon2 = cd.cross_end.lon
    
    hor = caculate_profile_wind(u,v,lat1=lat1, lat2=lat2, lon1=lon1, lon2=lon2)
    # dudz = (cal.first_derivative(hor,axis=0,x=z)).magnitude
    dudz = (cal.first_derivative(u,axis=0,x=z)).magnitude
    dvdz = (cal.first_derivative(v,axis=0,x=z)).magnitude

    # Ri = Nm2/(dudz**2)
    Ri = Nm2/(dudz**2+dvdz**2)
    da_ri = me2xr(Ri)

    return da_ri


def caculate_Nm(ds):
    """
    综合计算Brunt-vasaiila频率
    """
    da_rh = caculate_rh(ds)
    da_Nms = caculate_br_saturated(ds)
    da_Nmus = caculate_br_unsaturated(ds)
    ## 合并饱和和未饱和Bv
    index_s = xr.where(da_rh>=0.90, 1, 0)
    index_us = xr.where(da_rh<0.90, 1, 0)
    da_Nm = (da_Nms*index_s+da_Nmus*index_us)
    return da_Nm



def caculate_cross_wind(flnm, ds):
    # flnm = '../../data_original/wrfout/sod_all/wrfout_d03_2021-07-20_00:00:00'
    cd = CrossData(flnm)
    lat1 = cd.cross_start.lat
    lat2 = cd.cross_end.lat
    lon1 = cd.cross_start.lon
    lon2 = cd.cross_end.lon

    u = ds['u'].values
    v = ds['v'].values
    hor = caculate_profile_wind(u,v,lat1=lat1, lat2=lat2, lon1=lon1, lon2=lon2)
    hor = me2xr(hor)

    return hor

def caculate_scorer(da_Nm, ds):
    term1 = da_Nm/(ds['u']-9)**2
    z = ds['z']
    # term1
    term2 = 1/(ds['u']-9)*cal.second_derivative(ds['u'], axis=0, x=z)
    # term2
    da_scorer = me2xr(term1.values-term2.values)
    da_scorer_gradient = me2xr(cal.first_derivative(da_scorer, axis=0, x=z))
    return da_scorer, da_scorer_gradient

def get_ds(flnm='../../data_original/wrfout/sod_all/wrfout_d03_2021-07-20_00:00:00'):
    # flnm = '../../data_original/wrfout/sod_all/wrfout_d03_2021-07-20_00:00:00'
    wrfin = nc.Dataset(flnm)
    ds = xr.Dataset()
    ds['u'] = wrf.getvar(wrfin, 'ua')
    ds['v'] = wrf.getvar(wrfin, 'va')
    ds['w'] = wrf.getvar(wrfin, 'wa')
    ds['z'] = wrf.getvar(wrfin, 'z')
    ds['temp'] = wrf.getvar(wrfin, 'temp', units='degC')
    ds['td'] = wrf.getvar(wrfin, 'td', units='degC')
    ds['pressure'] = wrf.getvar(wrfin, 'pressure')
    return ds


def draw_cross_section():
    flnm='../../data_original/wrfout/sod_all/wrfout_d03_2021-07-20_00:00:00'
    ds = get_ds(flnm)
    da_Nm = caculate_Nm(ds)
    hor = caculate_cross_wind(flnm, ds)
    da_ri = caculate_ri(da_Nm, ds)

    ## 计算剖面数据
    # flnm = '../../data_original/wrfout/sod_all/wrfout_d03_2021-07-20_00:20:00'
    cd = CrossData(flnm)
    var =  wrf.getvar(cd.ncfile, 'temp')  # 直接传变量, 不要传文件

    da_Nm.attrs = var.attrs
    da_Nm_cross = cd.get_vcross(da_Nm)

    da_ri.attrs = var.attrs
    da_ri_cross = cd.get_vcross(da_ri)

    hor.attrs = ds['u'].attrs
    da_hor_cross= cd.get_vcross(hor)

    ter_line = cd.get_ter()

    ds['Nm'] =  da_Nm
    ds['hor'] = hor
    ds['ter'] = ter_line
    




    cm = 1/2.54
    fig = plt.figure(figsize=(8*cm,6*cm),dpi=300)
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    dv = DrawVertical(fig, ax)
    # dv.colordict=['gray']
    dv.colordict=['#CFCFCF']
    dv.colorlevel=[-5000000, 0.25]
    dv.colorticks = dv.colorlevel[1:-1]

    dv.draw_contourf(da_ri_cross, ter_line, levels=dv.colorlevel)
    dv.draw_contour(da_hor_cross, levels=[9,], colors='red')
    # dv.draw_contourf(da_Nm_cross, ter_line,levels=[-0.0001, 0,0.0001, 0.0002])
    # contour_levels = [0.00025,]# 0.00008,0.00016, 0.00024]
    dv.draw_contour(da_Nm_cross, levels=[0, 0.0001], colors=['black',],linestyles=['-', '--'])
    dv.ax.set_ylim(0, 7000)
    dv.fig.savefig('/home/fengx20/project/sod/picture/cross_section.png')

def draw_vertical_profile():

    flnm='../../data_original/wrfout/sod_all/wrfout_d03_2021-07-20_00:00:00'
    ds = get_ds(flnm)
    da_Nm = caculate_Nm(ds)
    hor = caculate_cross_wind(flnm, ds)
    da_ri = caculate_ri(da_Nm, ds)
    da_scorer, d2 = caculate_scorer(da_Nm, ds)

    da_Nm_profile = caculate_average_wrf(da_Nm)
    da_wind_profile = caculate_average_wrf(hor)
    da_ri_profile= caculate_average_wrf(da_ri)
    da_scorer_profile= caculate_average_wrf(d2) # 直接画它的梯度
    zz = caculate_average_wrf(ds['z'])

    # da_Nm_us_vertical = caculate_average_wrf(da_Nmus)  # un saturate
    # da_Nm_s_vertical = caculate_average_wrf(da_Nms)
    cm = 1/2.54
    fig = plt.figure(figsize=(8*cm,6*cm),dpi=300)
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    c = np.array([255,140,46])/255
    ax.plot(da_Nm_profile*10**5,zz, label=r'$N_m^2$', color=c)
    ax.plot(da_wind_profile,zz, label=r'$U$', color='black')
    ax.plot(da_scorer_profile*10**6,zz, label=r'$\frac{\partial l^2}{\partial z}$', color='blue')
    # ax.plot(d2*10**6,zz)
    # ax.plot(da_ri_profile,zz)
    ax.set_ylim(0,7000)
    ax.set_xlim(-5, 18)
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Height (m)')
    ax.legend(edgecolor='white', fontsize=10)
    fig.savefig('/home/fengx20/project/sod/picture/vertical_profile.png')

def draw_obs():
    flnm = '/home/fengx20/project/sod/data_combine/micaps_sounding_station_all.nc'
    ds = xr.open_dataset(flnm)
    # ds1 = ds.sel(station='nanyang').isel(time=10)
    ds1 = ds.sel(station='nanyang').sel(time='2021-07-20 00')
    # ds1 = ds.sel(station='zhengzhou').sel(time='2021-07-20 00')
    ds2 = ds1.dropna(dim='vertical')
    ds2 = ds2.rename({'height':'z'})

    Nm_obs = caculate_Nm(ds2)
    scorer, scorerT = caculate_scorer(Nm_obs,ds2)
    flnm = '../../data_original/wrfout/sod_all/wrfout_d03_2021-07-20_00:20:00'
    hor_wind = caculate_cross_wind(flnm, ds2)
    rh = caculate_rh(ds2)
    da = xr.DataArray(scorerT*10**6)
    db = xr.where((da<-10)|(da>10),np.nan, da)
    dc = db.interpolate_na(dim='dim_0', method='quadratic')
    dc.plot()

    def moving_average(interval, windowsize):
        window = np.ones(int(windowsize)) / float(windowsize)
        re = np.convolve(interval, window, 'same')
        return re

if __name__ == "__main__":
    draw_cross_section()
    # draw_vertical_profile()