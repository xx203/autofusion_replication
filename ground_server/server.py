import time
import yaml # 我们之后会用到配置
import os   # 我们之后会用到路径处理

# 未来会从同级目录导入其他模块
# from .initialization import Initializer
# from .pose_graph_optimizer import PoseGraphOptimizer
# from .map_fusion import MapFuser
# from vgmn.inference import ... # 还需要从 vgmn 导入

class GroundServer:
    """
    代表地面服务器。
    负责接收来自 Agent 的数据，运行 V-GMN，执行初始化、优化和地图融合。
    """
    def __init__(self, config):
        """
        初始化 Ground Server。

        Args:
            config (dict): 从 yaml 文件加载的配置信息。
        """
        self.config = config
        print("[Ground Server] Initializing...")

        # --- 数据存储结构 ---
        # 用来存储从各个 Agent 接收到的关键帧信息
        # 格式: { agent_id: { kf_id: {'local_pose': T, 'timestamp': ts, 'image': img_or_feature}, ... }, ... }
        self.agent_keyframes = {}

        # 用来存储 VGMN 处理后的结果 (地理位置、特征描述符)
        # 格式: { agent_id: { kf_id: {'geo': geo_info, 'features': features}, ... }, ... }
        self.vgmn_results = {}

        # 用来存储计算出的 Agent 之间的相对位姿 (相对于参考 Agent)
        # 格式: { agent_id: T_ref_agent (4x4 numpy array), ... }
        self.relative_transforms = {}

        # 用来存储优化后的全局位姿
        # 格式: { (agent_id, kf_id): T_world_kf (4x4 numpy array), ... }
        self.global_poses = {}

        # --- 状态标志 ---
        self.initialization_done = False # 是否完成了初始的相对位姿估计
        self.reference_agent_id = None   # 参考 Agent 的 ID

        # --- 初始化其他组件 (未来实现) ---
        # self.vgmn_model = self.load_vgmn()
        # self.initializer = Initializer(config)
        # self.optimizer = PoseGraphOptimizer(config)
        # self.map_fuser = MapFuser(config)
        # self.communication_stub = self.setup_communication()

        print("[Ground Server] Initialized. Waiting for agent data...")

    def run(self):
        """
        服务器的主运行循环。
        (在这个基础版本中，它只是简单地等待，未来会处理消息)
        """
        print("[Ground Server] Starting main loop...")
        try:
            while True:
                # --- 未来在这里添加接收和处理 Agent 消息的逻辑 ---
                # received_message = self.communication_stub.receive()
                # if received_message:
                #     self.handle_agent_message(received_message)

                # --- 未来可能在这里周期性地触发优化或融合 ---
                # if self.initialization_done:
                #     self.run_optimization_maybe()
                #     self.run_map_fusion_maybe()

                # 简单地暂停一下，避免空循环占用 CPU
                time.sleep(0.5)

        except KeyboardInterrupt:
            # 允许通过 Ctrl+C 停止服务器
            print("\n[Ground Server] Received KeyboardInterrupt. Shutting down...")

    def handle_agent_message(self, message):
        """
        处理从 Agent 收到的消息。
        现在会存储 'keyframe' 类型的消息到 self.agent_keyframes。

        Args:
            message (dict): 包含消息类型和数据的字典。
        """
        msg_type = message.get('type')
        agent_id = message.get('agent_id')
        kf_id = message.get('kf_id') # Keyframe ID from the agent

        # 基本信息打印，保持不变
        print(f"[Ground Server] Received message: Type={msg_type}, Agent={agent_id}, KF_ID={kf_id}")

        if msg_type == 'keyframe':
            # --- 开始添加存储逻辑 ---

            # 1. 确保该 agent_id 在字典中有条目，如果没有则创建
            if agent_id not in self.agent_keyframes:
                self.agent_keyframes[agent_id] = {}
                print(f"[Ground Server] Registered new agent: {agent_id}")

            # 2. 获取关键帧信息 (可以根据需要选择存储哪些)
            kf_timestamp = message.get('timestamp')
            kf_local_pose = message.get('local_pose') # 这是个 numpy 数组
            # kf_image_or_feature = message.get('image') # 暂时还是 None

            # 3. 将关键帧信息存储到字典中
            if kf_id is not None: # 确保有关键帧 ID
                self.agent_keyframes[agent_id][kf_id] = {
                    'timestamp': kf_timestamp,
                    'local_pose': kf_local_pose,
                    # 'image_or_feature': kf_image_or_feature # 未来可以存储图像或特征
                }
                print(f"[Ground Server] Stored keyframe {kf_id} for agent {agent_id}.")
            else:
                 print(f"[Ground Server] Warning: Received keyframe message without kf_id from agent {agent_id}.")


            # --- 未来在这里添加运行 VGMN、尝试初始化的逻辑 ---
            # self.run_vgmn_on_keyframe(agent_id, kf_id, kf_image_or_feature)
            # if not self.initialization_done and self.check_initialization_readiness():
            #     self.run_initialization()

            # --- 存储逻辑结束 ---
        else:
            print(f"[Ground Server] Received unknown message type: {msg_type}")

    # --- 其他方法 (未来实现) ---
    # def load_vgmn(self): pass
    # def setup_communication(self): pass
    # def check_initialization_readiness(self): pass
    # def run_optimization_maybe(self): pass
    # def run_map_fusion_maybe(self): pass