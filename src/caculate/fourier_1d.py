#!/home/fengxiang/anaconda3/envs/wrfout/bin/python
# -*- encoding: utf-8 -*-
'''
Description:
一维傅里叶变换测试, 只测试和理解一维傅里叶变换
-----------------------------------------
Time             :2023/02/09 18:48:33
Author           :Forxd
Version          :1.0
'''


# %%
import numpy as np
import xarray as xr
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号

# %%
# 采样点选择1400个，因为设置的信号频率分量最高为600赫兹，根据采样定理知采样频率要大于信号频率2倍，
# 所以这里设置采样频率为1400赫兹（即一秒内有1400个采样点）
# 总共有1400个数据点, 采样频率设为1400HZ,  1s内获得这1400个点的数据，
# 而信号频率600HZ, 则是在1400个点中包含有600个波, 差不多2个点就有一个完整信号

N = 1400  # 设置1400个采样点, 1400个数据, 比如说每隔10分钟一个数据啊，这样
x = np.linspace(0, 1, N)  # 将0到1平分成1400份

# 设置需要采样的信号，频率分量有0，200，400和600, 还有个直流信号10
# 人为设置信号, 这个位相是一样的
y = 7 * np.sin(2 * np.pi * 200 * x) + 5 * np.sin(
    2 * np.pi * 400 * x) + 3 * np.sin(2 * np.pi * 600 * x) + 10  # 构造一个演示用的组合信号

plt.plot(x, y)
plt.title('原始波形')
plt.xlim(0, 0.1)
plt.show()
# %%

flnm = '/home/fengx20/project/LeeWave/data/output/cross_GWD3.nc'
ds = xr.open_dataset(flnm)

da = ds['wa_cross']
da1 = da.interpolate_na(dim='vertical', method='linear',  fill_value="extrapolate")
da2 = da1.interp(vertical=[5000], method='linear',  kwargs={"fill_value": "extrapolate"}).squeeze()
# %%
# da3 = da2.isel(time=20)
# val = da3.values
da3 = da2.isel(cross_line_idx=10)
val = da3.values
# da3
# val.shape
# %%

# N = 1400  # 设置1400个采样点, 1400个数据, 比如说每隔10分钟一个数据啊，这样
# x = np.linspace(0, 1, N)  # 将0到1平分成1400份
y = val
N = len(val)
x = np.linspace(0, 1, N)  # 将0到1平分成1400份

y
# x
plt.plot(x,y)


# %%
### 求幅度谱, 对应的是频域图像
## 用的是scipy的快速傅里叶变换，
# fft_y = fft(y)  # 使用快速傅里叶变换，得到的fft_y是长度为N的复数数组
fft_y = np.fft.fft(y)  
x = np.arange(N)  # 频率个数 （x的取值涉及到横轴的设置，这里暂时忽略，在第二节求频率时讲解）

"""
fft_y是复数
直流分量的振幅是np.abs(fft_y)/N, 即10
直流分量意外的振幅是np.abs(fft_y)/(N/2)
当前值/FFT采样点个数
"""
normalization_y = np.abs(fft_y)
normalization_y[0] /= 2
# plt.plot(normalization_y/N*2)
# plt.plot(normalization_y/N)
plt.plot(x, normalization_y/N*2, 'black')
plt.title('双边振幅谱(未求振幅绝对值)', fontsize=9, color='black')
# %%
# normalization_y[0]/N*2


# %%
# normalization_y[0] /= 2
# plt.xlim(0, N/2)
# %%
fig = plt.figure(figsize=(4,3),dpi=300)
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
# ax1.plot(x, y, 'black')
ax1.plot(x, normalization_y/N*2, 'black') # 重新转化为了振幅
# ax2.plot(1/x*N, normalization_y/N*2, 'black')  #  转化为了周期
# ax2.plot(1/x*N, normalization_y/N*2, 'black')  #  转化为了周期
ax2.plot(1/x*N, normalization_y/N*2, 'black')  #  转化为了周期
ax2.set_xlim(2, 10)
# ax2.set_ylim(0, 20)
# ax2.set_xticks(np.arange(2, 25, 2))
# ax1.set_xlim(0, N/2)
# ax2.set_xlim(0, N/2*144)


# %%
# x
x
# x
# 1/x
# x

# plt.show()
# %%
# 振幅，频率，相位
# %%
# sst
fig = plt.figure(figsize=(6,6),dpi=300)
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
# sst = da.values[50:150]
sst = y
sample_rate = len(sst)   # 假设有这样一个信号, 这个就是信号的个数
N = len(sst)
noised_sigs = sst
times = np.arange(noised_sigs.size)/sample_rate
ax1.plot(times*N, noised_sigs, color='orangered')
# ax1.set_title('Time DOmain')
# times
### 基于傅里叶变换，获取频域信息
freqs = np.fft.fftfreq(times.size, times[1]-times[0])
# %%
complex_array = np.fft.fft(noised_sigs)
pows = np.abs(complex_array)
# ax1.plot(freqs[freqs>0], pows[freqs>0])
T = 1/freqs*N
# A = pows/N*2
A = np.sqrt(pows)
# ax2.plot(1/freqs[freqs>0]*N, pows[freqs>0]/N*2)
# ax2.plot(1/freqs[freqs>0]*N, pows[freqs>0]/N*2)
ax2.plot(T[T>0],A[T>0])
ax2.set_title('Frequency Domain')
ax2.set_xlim(0,50)
ax3.set_xlim(0,50)
## 需要周期什么范围内的值
t1 = 0.0001
t2 = 100000
# t1 = 0.0001
# t2 = 24
f1 = N/t1
f2 = N/t2

## 将低频噪声去除掉
# 寻找能量最大的频率值
# fund_freq = freqs[pows.argmax()]
# fund_freq = freqs[pows.argmin()]
# where函数寻找那些需要抹掉的复数的索引
# noised_indices = np.where(freqs != fund_freq)
# noised_indices = np.where(freqs != 25)

## 留下这些
# f1
# f2

# noised_indices = np.where((freqs>f2) & (freqs<f1))
noised_indices = np.where((freqs<=f2) | (freqs>=f1))
print(noised_indices)
# noised_indices = np.where((freqs<23)&(freqs>25.0))
# noised_indices = np.where((freqs>24)&(freqs<25))
# print(noised_indices)
# noised_indices = np.where(freqs == fund_freq)
# 复制一个复数数组的副本，避免污染原始数据
filter_complex_array = complex_array.copy()
# filter_complex_array[noised_indices] = 0
#### 6~9小时周期
# filter_complex_array[17:] = 0
# filter_complex_array[:11] = 0
#### 17~30小时周期
# filter_complex_array[6:] = 0
# filter_complex_array[:3] = 0

# filter_complex_array[17:] = 0
filter_complex_array[11:17] = 0
filter_pows = np.abs(filter_complex_array)
# ax3.plot(freqs[freqs >= 0], filter_pows[freqs >= 0], c='dodgerblue', label='Filter')
ax3.plot(T[T >= 0], np.sqrt(filter_pows[freqs >= 0]), c='dodgerblue', label='Filter')


filter_sigs = np.fft.ifft(filter_complex_array).real
# ax4.plot(times[:178], filter_sigs[:178], c='hotpink', label='Filter')
# ax4.plot(times*N, filter_sigs*2+4, c='hotpink', label='Filter')
ax4.plot(times*N, filter_sigs, c='hotpink', label='Filter')
ax4.set_xlim(30,70)
# ax4.plot(T[:178], filter_sigs[:178], c='hotpink', label='Filter')
# %%
# print(noised_sigs)
# filter_sigs.max()
# sst.max()
# val
plt.plot(val)
# filter_complex_array.shape
# freqs.min()
# fund_freq
# pows