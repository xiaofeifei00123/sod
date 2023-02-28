# -*- encoding: utf-8 -*-
'''
Description:
设置共用的位置信息等
-----------------------------------------
Time             :2023/02/12 16:23:49
Author           :Forxd
Version          :1.0
'''

import cartopy.crs as ccrs

# from wrf import CoordPair
class Common():
    def __init__(self, ):

        self.areaA = {
            'lat1':33.5,
            'lat2':33.8,
            'lon1':112.4,
            'lon2':112.8,
        }        
        ## 模式降水核心区
        self.areaB = {
            'lat1':34.4,
            'lat2':34.8,
            'lon1':113.0,
            'lon2':113.4,
        }        
        self.areaC = {
            'lat1':35.3,
            'lat2':35.8,
            'lon1':113.5,
            'lon2':114.2,
        }        
        ## 观测降水核心区
        self.areaE = {
            'lat1':34.4,
            'lat2':34.8,
            'lon1':113.2,
            'lon2':113.6,
        }        
        ## 整个河南区域
        self.areaall = {
            'lat1':32.0,
            'lat2':36.5,
            'lon1':110.5,
            'lon2':116,
        }        

        
        ## 波动的一个区域
        self.area_bo= {
            'lat1':33.4,
            'lat2':33.7,
            'lon1':112.6,
            'lon2':112.9,
        }        
        ## 进行地形平滑的区域
        self.area_terrain = {
            'lat1':33.0,
            'lat2':34.1,
            'lon1':111.7,
            'lon2':112.8,
            # 'lon2':113.5,
        }        

        self.station_line = {
                'A': {
                    'abbreviation':'A',
                    'lat': 33.6,
                    'lon': 112.1,
                },
                'B': {
                    'abbreviation':'B',
                    'lat': 34.5,
                    'lon': 112.9,
                },
                'C': {
                    'abbreviation':'C',
                    'lat': 35.15,
                    'lon': 113.5,
                },
                'D': {
                    'abbreviation':'D',
                    'lat': 35.75,
                    'lon': 114.0,
                },
            }

        self.station_sta = {
                'ZhengZhou': {
                    'abbreviation':'郑州',
                    'lat': 34.76,
                    'lon': 113.65
                },
                'NanYang': {
                    'abbreviation':'南阳',
                    'lat': 33.1,
                    'lon': 112.49,
                },
                'LuShi': {
                    'abbreviation':'卢氏',
                    'lat': 34.08,
                    'lon': 111.07,
                },
            }

        self.station_zz = {
                'ZhengZhou': {
                    'abbreviation':'Zhengzhou',
                    'lat': 34.76,
                    'lon': 113.65
                },
            }
        self.cross_start = [111.5, 33]
        self.cross_end = [113.5, 34.9]
        

        ## 沿风向过伏牛山和嵩山的切线
        self.cross = {
            'start':{
                'lon':self.cross_start[0],
                'lat':self.cross_start[1],
            },
            'end':{
                'lon':self.cross_end[0],
                'lat':self.cross_end[1],
            }
        }

        self.map_dic_d03 = {
            'proj':ccrs.PlateCarree(),
            'extent':[107, 120, 29, 39],
            'extent_interval_lat':1,
            'extent_interval_lon':1,
        }
        self.map_dic_d04 = {
            'proj':ccrs.PlateCarree(),
            'extent':[110.5, 116, 32, 36.5],
            'extent_interval_lat':1,
            'extent_interval_lon':1,
        }
        # self.cross_start = [112.2, 32.5]
        # self.cross_end = [113.7, 35.5]
        # self.cross_start_coord = CoordPair(lat=33, lon=111.5)
        # self.cross_end_coord = CoordPair(lat=36, lon=114.3)