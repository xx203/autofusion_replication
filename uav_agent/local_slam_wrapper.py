import numpy as np
import time # 我们会用到它来模拟时间戳

class LocalSLAMWrapper:
    """
    一个【模拟】的 SLAM/VO 系统。
    它不进行真正的视觉计算，只是根据预设规则生成假的位姿和关键帧。
    """
    def __init__(self, agent_id=0):
        print(f"[SLAM Wrapper {agent_id}] Initializing Simulated SLAM...")
        self.agent_id = agent_id
        # 初始位姿：单位矩阵 (表示在原点，无旋转)
        self.current_pose = np.eye(4) # 4x4 SE(3) 齐次变换矩阵
        self.keyframes = {} # 存储关键帧信息 {id: {'pose': pose_matrix, 'timestamp': ts}}
        self.keyframe_count = 0
        self.last_keyframe_time = time.time()
        self.frame_index = 0 # 模拟处理的帧序号

        # 模拟参数 (可以随意调整)
        self.sim_translation_speed = 0.5 # 米/帧 (模拟每次移动的距离)
        self.sim_rotation_speed = 0.02 # 弧度/帧 (模拟每次旋转的角度)
        self.keyframe_interval_seconds = 1.0 # 大约每隔多少秒生成一个关键帧

    def process_image(self, image, timestamp):
        """
        【模拟】处理一帧图像。
        在这个模拟版本中，我们忽略 'image' 参数。
        根据时间戳和帧序号来更新位姿，并决定是否生成关键帧。

        Args:
            image: 实际应用中是图像数据，这里忽略。
            timestamp (float): 当前帧的时间戳 (秒)。

        Returns:
            tuple or None: 如果生成了新的关键帧，返回 (kf_id, kf_info)，否则返回 None。
                           kf_info 是包含 'pose' 和 'timestamp' 的字典。
        """
        self.frame_index += 1

        # 1. 模拟位姿更新 (简单的向前移动和绕 Z 轴旋转)
        # 计算平移增量 (假设沿 x 轴移动)
        delta_translation = np.array([self.sim_translation_speed, 0, 0])
        # 计算旋转增量 (绕 z 轴)
        delta_rotation_angle = self.sim_rotation_speed
        delta_rotation_matrix = np.array([
            [np.cos(delta_rotation_angle), -np.sin(delta_rotation_angle), 0, 0],
            [np.sin(delta_rotation_angle), np.cos(delta_rotation_angle),  0, 0],
            [0,                             0,                              1, 0],
            [0,                             0,                              0, 1]
        ])
        # 创建增量变换矩阵
        delta_transform = np.eye(4)
        delta_transform[:3, :3] = delta_rotation_matrix[:3, :3] # 应用旋转部分
        delta_transform[:3, 3] = delta_translation           # 应用平移部分

        # 更新当前位姿: T_world_new = T_world_old * T_old_new
        # 注意：这里的 delta_transform 是在当前帧坐标系下的变换 T_old_new
        self.current_pose = self.current_pose @ delta_transform

        # 打印一些模拟信息 (可选)
        # if self.frame_index % 20 == 0: # 每 20 帧打印一次
        #    print(f"[SLAM Wrapper {self.agent_id}] Frame {self.frame_index}, Current Pose:\n{self.current_pose[:3,:]}")

        # 2. 决定是否生成关键帧 (基于时间间隔)
        if timestamp - self.last_keyframe_time >= self.keyframe_interval_seconds:
            kf_id = self.keyframe_count
            kf_pose = self.current_pose.copy() # 复制当前的位姿作为关键帧位姿
            kf_timestamp = timestamp
            self.keyframes[kf_id] = {'pose': kf_pose, 'timestamp': kf_timestamp}

            self.keyframe_count += 1
            self.last_keyframe_time = timestamp
            print(f"[SLAM Wrapper {self.agent_id}] ---- New Keyframe Generated ---- ID: {kf_id} at time {kf_timestamp:.2f}")

            # 实际应用中，这里还会有关联的图像、特征点等信息
            kf_info_to_send = {'pose': kf_pose, 'timestamp': kf_timestamp}
            return kf_id, kf_info_to_send # 返回关键帧 ID 和信息

        return None # 没有生成新的关键帧

    def get_current_pose(self):
        """获取当前【模拟】的位姿"""
        return self.current_pose

    # --- 在这个模拟版本中，我们暂时不需要 pose correction 功能 ---
    # def update_pose(self, correction_transform):
    #      pass