import yaml # 用于读取 yaml 文件
import os   # 用于处理文件路径
import numpy as np

# 导入我们自己写的 utils 模块
# Python 会根据文件结构自动找到 utils 包
from utils import geo, transformations

print("--- 开始测试项目设置 ---")

# 1. 测试读取配置文件
config_file_path = os.path.join("configs", "params.yaml") # 构建配置文件的路径
print(f"尝试读取配置文件: {config_file_path}")

try:
    with open(config_file_path, 'r', encoding='utf-8') as f: # <--- 在这里添加 encoding='utf-8'
        config = yaml.safe_load(f) # 使用 PyYAML 读取文件内容
    print("成功读取配置文件:")
    print(config) # 打印读取到的配置内容

    # 访问配置项
    print(f"模拟的无人机数量: {config.get('simulation', {}).get('num_uavs', '未找到')}")
    print(f"地图融合体素大小: {config.get('map_fusion', {}).get('voxel_size', '未找到')}")

except FileNotFoundError:
    print(f"错误：找不到配置文件 {config_file_path}")
except Exception as e:
    print(f"读取配置文件时发生错误: {e}")

print("\n--- 测试调用 utils 模块函数 ---")

# 2. 测试调用 utils.geo 中的函数
try:
    lat1, lon1 = 40.7128, -74.0060 # 纽约市 经纬度 (示例)
    lat2, lon2 = 34.0522, -118.2437 # 洛杉矶 经纬度 (示例)
    distance = geo.haversine_distance(lat1, lon1, lat2, lon2)
    print(f"使用 geo.haversine_distance 计算纽约到洛杉矶的距离: {distance / 1000:.2f} 公里")

    x, y, z = 210270, 104690, 18 # 论文图4中的一个瓦片坐标示例
    tile_lat, tile_lon = geo.tile_to_latlon(x, y, z)
    print(f"使用 geo.tile_to_latlon 将瓦片 ({x},{y},{z}) 转为经纬度: ({tile_lat:.6f}, {tile_lon:.6f})")

except Exception as e:
    print(f"调用 utils.geo 函数时发生错误: {e}")

# 3. 测试调用 utils.transformations 中的函数 (只是示例)
try:
    point = np.array([1.0, 2.0, 3.0])
    # 创建一个简单的平移变换矩阵 (向x轴平移5个单位)
    transform_matrix = np.array([
        [1, 0, 0, 5],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    transformed_point = transformations.example_transform_function(point, transform_matrix)
    print(f"使用 transformations.example_transform_function 变换点 {point}: {transformed_point}")
except NameError:
     # 我们在 transformations.py 中没有导入 numpy as np，所以这里会报错
     # 这也测试了模块导入的隔离性。我们稍后会修复它。
     print("调用 transformations 函数时发生 NameError (预期之中，因为 numpy 未在 transformations.py 中导入)")
except Exception as e:
    print(f"调用 utils.transformations 函数时发生错误: {e}")


print("\n--- 测试结束 ---")