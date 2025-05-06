import numpy as np

# 这里我们会放一些处理3D旋转和平移的函数
# 例如，计算旋转矩阵、四元数转换、坐标系变换等

def example_transform_function(point, transform_matrix):
    """
    一个示例函数，说明如何应用变换矩阵。
    这不是论文里的具体内容，只是个例子。

    Args:
        point (np.array): 3D 点坐标，例如 np.array([x, y, z])
        transform_matrix (np.array): 4x4 的齐次变换矩阵

    Returns:
        np.array: 变换后的 3D 点坐标
    """
    if point.shape[0] != 3:
        raise ValueError("输入点必须是 3D 坐标")
    if transform_matrix.shape != (4, 4):
        raise ValueError("变换矩阵必须是 4x4")

    # 将 3D 点转换为齐次坐标 (添加 1)
    point_h = np.append(point, 1)

    # 应用变换矩阵
    transformed_point_h = transform_matrix @ point_h

    # 将结果转回 3D 坐标 (去掉最后的 1，并确保是 3 维)
    transformed_point = transformed_point_h[:3] / transformed_point_h[3] # 考虑尺度因子

    return transformed_point

# --- 未来可能添加的函数 ---
# def get_rotation_matrix(axis, angle): pass
# def get_quaternion_from_matrix(matrix): pass
# def multiply_transforms(T1, T2): pass
# def invert_transform(T): pass

print("utils/transformations.py loaded") # 打印一条消息确认文件被加载了