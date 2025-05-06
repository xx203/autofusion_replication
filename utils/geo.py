import numpy as np
import math

# 地球平均半径（米），用于 Haversine 计算
EARTH_RADIUS_METERS = 6371000

def tile_to_latlon(x, y, z):
    """
    将瓦片坐标 (x, y) 和层级 (z) 转换为经纬度。
    基于论文中的描述（公式1的一部分）。
    参考: https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Python
    """
    n = 2.0 ** z
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg

def haversine_distance(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    """
    使用 Haversine 公式计算两个经纬度点之间的球面距离（米）。
    对应论文中的公式 6 的核心计算。
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1_deg, lon1_deg, lat2_deg, lon2_deg])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a)) # 或者 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = EARTH_RADIUS_METERS * c
    return distance

# --- 未来可能添加的函数 ---
# def utm_to_latlon(easting, northing, zone_number, zone_letter): pass
# def calculate_scale_from_vo_and_geo(vo_coords1, vo_coords2, geo_coords1, geo_coords2):
#    # 对应论文公式 5, 但需要小心验证公式含义
#    pass

print("utils/geo.py loaded") # 打印一条消息确认文件被加载了