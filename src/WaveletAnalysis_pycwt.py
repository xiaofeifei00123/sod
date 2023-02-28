# %%
import numpy as np
import matplotlib.pyplot as plt
import pycwt as wavelet
from pycwt.helpers import find
from matplotlib.pyplot import MultipleLocator
import xarray as xr
plt.style.use('~/mypy/baobao/my.mplstyle')
# %%
def get_data_rain():
    """
    返回sst, time, da
    """
    def caculate_area_mean_obs(da,area):
        mask = (
            (da.coords['lat']>area['lat1'])
            &(da.coords['lat']<area['lat2'])
            &(da.coords['lon']<area['lon2'])
            &(da.coords['lon']>area['lon1'])
        )
        aa = xr.where(mask, 1, np.nan)
        db = da*aa
        dsr = db.mean(dim=['lat', 'lon'])
        return dsr
    # area = {
    #     'lat1':34.4,
    #     'lat2':34.8,
    #     'lon1':113.2,
    #     'lon2':113.6,
    #     }        
    area = {
        'lat1':34.5,
        'lat2':34.6,
        'lon1':113.2,
        'lon2':113.6,
        }        
    # area = {
    #     'lat1':32,
    #     'lat2':36.5,
    #     'lon1':110.5,
    #     'lon2':116,
    #     }        
    flnm_obs = '/home/fengx20/project/HeNan/Data/rain_obs.nc'
    ds_obs = xr.open_dataset(flnm_obs)
    # gd = GetData()
    ds_obs_mean  = caculate_area_mean_obs(ds_obs, area)
    sst = ds_obs_mean['PRCP'].values
    tt = ds_obs_mean.time.values
    da = ds_obs_mean['PRCP']
    return sst, tt, da




# def main():


sst, time, da = get_data_rain()
# %%
# dat = sst
# dat.max()




# 从网页获取数据
url = 'http://paos.colorado.edu/research/wavelets/wave_idl/nino3sst.txt'
dat = np.genfromtxt(url, skip_header=19)
print(dat.shape)

# type(dat)
# type(sst)
# ddt.at1 = dat
# x_label = x_label.dt.strftime('%d/%H')

# 504/4
# 1871+126
# %%
dat = sst
#  %%
# time.strftime('%Y')
# pd.strftime(time[0])
# import pandas as pd
# ttt = pd.Series(time)
# ttt[0].strftime('%Y%m')
xlabel = da.time.dt.strftime('%d/%H')

# %%




title = 'Hourly Precipitaiton'   # 标题
label = 'Precipitation'                       # 标签 
units = 'mm'                            # 单位
# t0 = 1871.0                               # 开始的时间，以年为单位
t0 = 0
# dt = 0.25                                 # 采样间隔，以年为单位
dt=1

N = dat.size                              # 时间序列的长度
t = np.arange(0, N) * dt + t0             # 构造时间序列数组


# ## 数据预处理（去趋势、标准化）


p = np.polyfit(t - t0, dat, 1)               # 线性拟合
dat_notrend = dat - np.polyval(p, t - t0)    # 去趋势
std = dat_notrend.std()                      # 标准差
var = std ** 2                               # 方差
dat_norm = dat_notrend / std                 # 标准化


# ## 选择小波基函数
# mother = wavelet.Morlet(6)      # Monther Wavelet: Morlet 
mother = wavelet.Morlet(6)      # Monther Wavelet: Morlet 
s0 = 2 * dt                     # Starting scale, in this case 2 * 0.25 years = 6 months
# s0 = dt                     # Starting scale, in this case 2 * 0.25 years = 6 months
dj = 0.25                       # Twelve sub-octaves per octaves
J = 7 / dj                      # Seven powers of two with dj sub-octaves
alpha, _, _ = wavelet.ar1(dat)  # Lag-1 autocorrelation for red noise
print(alpha)
# ## 计算小波变换(wave)和逆小波变换(iwave)


wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, J, mother)
iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std


# ## 计算小波能量谱
# 能量谱也叫能量谱密度，能量谱密度描述了信号或时间序列的能量如何随频率分布。能量谱是原信号傅立叶变换的平方。

power = np.power(np.abs(wave),2)
fft_power = np.power(np.abs(fft),2)
period = 1 / freqs
#power /= scales[:,None]


# ## 计算小波功率谱
# 功率谱是功率谱密度函数，它定义为单位频带内的信号功率，是针对功率信号来说的。
# 求功率谱就有了两种方法，分别叫做直接法和相关函数法：
# **1、(傅立叶变换的平方)/(区间长度)；2、自相关函数的傅里叶变换。**

signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                                        significance_level=0.95,
                                        wavelet=mother)
sig95 = np.ones([1, N]) * signif[:, None]
sig95 = power / sig95



glbl_power = power.mean(axis=1)
dof = N - scales                   # Correction for padding at edges
glbl_signif, tmp = wavelet.significance(var, dt, scales, 1, alpha,
                                        significance_level=0.95, dof=dof,
                                        wavelet=mother)


# ## 计算小波方差


sel = find((period >= 2) & (period < 8))
Cdelta = mother.cdelta
scale_avg = (scales * np.ones((N, 1))).transpose()
scale_avg = power / scale_avg  # As in Torrence and Compo (1998) equation 24
scale_avg = var * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
scale_avg_signif, tmp = wavelet.significance(var, dt, scales, 2, alpha,
                                            significance_level=0.95,
                                            dof=[scales[sel[0]],
                                                scales[sel[-1]]],
                                            wavelet=mother)


## 绘制小波分析结果


# Prepare the figure
fig = plt.figure(figsize=(11, 8))

# First sub-plot, the original time series anomaly and inverse wavelet transform.
ax = plt.axes([0.1, 0.75, 0.65, 0.2])
# ax.plot(t, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
# ax.plot(t, dat, 'k', linewidth=1.5)
ax.plot(xlabel, iwave, '-', linewidth=1, color=[0.5, 0.5, 0.5])
ax.plot(xlabel, dat, 'k', linewidth=1.5)
ax.set_xticks(xlabel[::12], rotation=30)
ax.set_title('a) {}'.format(title))
ax.set_ylabel(r'{} [{}]'.format(label, units))


# Second sub-plot, the normalized wavelet power spectrum and significance
# level contour lines and cone of influece hatched area. Note that period
# scale is logarithmic.
bx = plt.axes([0.1, 0.37, 0.65, 0.28], sharex=ax)
levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
# bx.contourf(t, period, power, levels, extend='both', cmap=plt.cm.viridis)
bx.contourf(t, np.log2(period), np.log2(power), np.log2(levels), extend='both', cmap=plt.cm.viridis)
# bx.set_ylim(0, 10)

bx.contourf(t, np.log2(period), np.log2(power), np.log2(levels), extend='both', cmap=plt.cm.viridis)
extent = [t.min(), t.max(), 0, max(period)]

bx.contour(t, np.log2(period), sig95, [-99, 1], colors='k', linewidths=1, extent=extent)
bx.fill(np.concatenate([t, t[-1:] + dt, t[-1:] + dt,t[:1] - dt, t[:1] - dt]),
        np.concatenate([np.log2(coi), [1e-9], np.log2(period[-1:]),np.log2(period[-1:]), [1e-9]]),
        'k', alpha=0.3, hatch='x')
bx.set_title('b) {} Wavelet Power Spectrum ({})'.format(label, mother.name))
bx.set_ylabel('Period (Hours)')
Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),np.ceil(np.log2(period.max())))
bx.set_yticks(np.log2(Yticks))
yt = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
# bx.set_yticks(np.log2(yt))
bx.set_yticklabels(Yticks)
bx.set_ylim(0, np.log2(16))


# Third sub-plot, the global wavelet and Fourier power spectra and theoretical
# noise spectra. Note that period scale is logarithmic.
cx = plt.axes([0.77, 0.37, 0.2, 0.28], sharey=bx)
cx.plot(glbl_signif, np.log2(period), 'k--')
cx.plot(var * fft_theor, np.log2(period), '--', color='#cccccc')
cx.plot(var * fft_power, np.log2(1/fftfreqs), '-', color='#cccccc',linewidth=1)
cx.plot(var * glbl_power, np.log2(period), 'k-', linewidth=1.5)
cx.set_title('c) Global Wavelet Spectrum')
cx.set_xlabel(r'Power [({})^2]'.format(units))
# cx.set_xlim([0, 6])
cx.set_xlim([0, 100])
cx.set_ylim(np.log2([period.min(), period.max()]))
cx.set_yticks(np.log2(Yticks))
cx.set_yticklabels(Yticks)
plt.setp(cx.get_yticklabels(), visible=False)

# cx.set_ylim(0, np.log2(16))
# cx.set_ylim(2, np.log2(32))

# Fourth sub-plot, the scale averaged wavelet spectrum.
dx = plt.axes([0.1, 0.07, 0.65, 0.2], sharex=ax)
dx.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1.)
dx.plot(t, scale_avg, 'k-', linewidth=1.5)
dx.set_title('d) scale-averaged power')
dx.set_xlabel('Time (date/hour)')
dx.set_ylabel(r'Average variance [{}]'.format(units))
ax.set_xlim([t.min(), t.max()])
plt.savefig('wavelet_analysis.jpg',dpi=600)
plt.show()


# %%
# ## 绘制小波方差
Cdelta = mother.cdelta
scale_avg = (scales * np.ones((N, 1))).transpose()
scale_avg = power / scale_avg  # As in Torrence and Compo (1998) equation 24
scale_avg = var * dj * dt / Cdelta * scale_avg[sel, :].sum(axis=0)
scale_avg_signif, tmp = wavelet.significance(var, dt, scales, 2, alpha,
                                            significance_level=0.95,
                                            dof=[scales[0],scales[-1]],
                                            wavelet=mother)
fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(111)
ax.plot(t-t0, scale_avg, 'k-')
ax.axhline(scale_avg_signif, color='k', linestyle='--', linewidth=1.)
ax.set_xlim(0,126)
ax.grid()
x_major_locator=MultipleLocator(5)
ax.xaxis.set_major_locator(x_major_locator)

# %%
# period
wave.real
# %%
## 绘制小波系数
fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(111)
cf = plt.contourf(t, np.log2(period), np.log2(wave.real), 
                np.arange(-4,4,0.5), extend='both', cmap=plt.cm.jet)
cf = plt.contourf(t, period, wave.real, 
                extend='both', cmap=plt.cm.jet)
#plt.clabel(cf,colors='k')
Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())),np.ceil(np.log2(period.max())))
# ax.set_yticks(np.log2(Yticks))
# ax.set_yticklabels(Yticks)
ax.set_ylim(0, 7)
ax.set_ylabel('Period (years)')
plt.colorbar()


# 参考资料：
# 1、[信号频域分析方法的理解（频谱、能量谱、功率谱、倒频谱、小波分析）](https://zhuanlan.zhihu.com/p/34989414)
# 2、[形象易懂讲解算法I——小波变换](https://zhuanlan.zhihu.com/p/22450818)
# 3、[PyCWT官方文档](https://pycwt.readthedocs.io/en/latest/index.html)
# 4、[初学应用举例（小波分析，实部画图，小波方差画图及小波模画图等）](http://bbs.06climate.com/forum.php?mod=viewthread&tid=11368)

# %%
# sst
# if __name__ == '__main__':
# main()

# %%