from .local_slam_wrapper import LocalSLAMWrapper
import time
import numpy as np # 只是为了示例消息里包含 numpy array

class UAVAgent:
    """
    代表一个无人机 Agent。
    它包含一个 SLAM 系统（现在是模拟的），并负责处理数据和【直接调用】服务器的方法。
    """
    def __init__(self, agent_id, config, server_instance): # <--- 添加 server_instance 参数
        """
        初始化 Agent。

        Args:
            agent_id (int): Agent 的唯一标识符。
            config (dict): 从 yaml 文件加载的配置信息。
            server_instance (GroundServer): 对 GroundServer 实例的引用。
        """
        self.agent_id = agent_id
        self.config = config
        self.server = server_instance # <--- 保存 Server 实例的引用
        print(f"[Agent {self.agent_id}] Initializing...")

        # 初始化 SLAM 系统 (现在使用模拟的 Wrapper)
        self.local_slam = LocalSLAMWrapper(agent_id=self.agent_id)

        if self.server is None:
             print(f"[Agent {self.agent_id}] Warning: No server instance provided.")


    def run_step(self, image=None, timestamp=None):
        """
        执行 Agent 的一个时间步。
        处理传感器数据（现在是模拟的），运行 SLAM，并将关键帧信息【直接发送】给 Server。

        Args:
            image: 图像数据 (模拟时忽略)。
            timestamp: 时间戳 (模拟时使用)。

        Returns:
            tuple or None: 如果本地 SLAM 生成了关键帧，返回 (kf_id, kf_info)，否则返回 None。
        """
        if timestamp is None:
            timestamp = time.time() # 如果没提供时间戳，就用当前时间

        # 1. 将数据喂给本地 SLAM 处理
        keyframe_result = self.local_slam.process_image(image, timestamp)

        # 2. 如果 SLAM 生成了关键帧，将其发送给 Server
        if keyframe_result:
            kf_id, kf_info = keyframe_result
            print(f"[Agent {self.agent_id}] Processed Keyframe {kf_id}.")
            # 打印位姿信息的部分可以移到这里或者保留在 SLAM Wrapper 里

            # --- 直接调用 Server 的方法来“发送”消息 ---
            if self.server:
                message = {
                   'type': 'keyframe',
                   'agent_id': self.agent_id,
                   'kf_id': kf_id,
                   'timestamp': kf_info['timestamp'],
                   'local_pose': kf_info['pose'], # numpy 数组
                   'image': None # 实际应用中会发送图像或特征，这里是 None
                }
                # 直接调用 handle_agent_message 方法
                self.server.handle_agent_message(message)
                print(f"[Agent {self.agent_id}] 'Sent' Keyframe {kf_id} to server.")
            else:
                print(f"[Agent {self.agent_id}] No server to send keyframe {kf_id} to.")


            return kf_id, kf_info # 将关键帧信息也返回给调用者

        # 3. (未来) 检查并处理来自服务器的消息 (在这个模拟中 Server 不会主动发消息)
        # server_messages = self.communication_stub.receive()
        # self.handle_server_messages(server_messages)

        return None # 这个时间步没有生成关键帧

    # --- 未来实现 ---
    # def handle_server_messages(self, messages):
    #    pass