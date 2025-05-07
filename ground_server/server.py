import time
import yaml # 我们之后会用到配置
import os   # 我们之后会用到路径处理

# 从同级目录导入其他模块
# from .initialization import Initializer
# from .pose_graph_optimizer import PoseGraphOptimizer
# from .map_fusion import MapFuser

# 新增导入 VVVV
from vgmn.inference import load_vgmn_model, run_vgmn_inference # <--- 导入 VGMN 推理函数

class GroundServer:
    def __init__(self, config):
        self.config = config
        print("[Ground Server] Initializing...")

        self.agent_keyframes = {}
        self.vgmn_results = {} # <--- 我们会把 VGMN 结果存这里
        self.relative_transforms = {}
        self.global_poses = {}
        self.initialization_done = False
        self.reference_agent_id = None

        # --- 初始化 VGMN 模型 --- VVVV
        print("[Ground Server] Attempting to load VGMN model...")
        self.vgmn_model = load_vgmn_model(self.config) # <--- 加载模型
        if self.vgmn_model:
            print("[Ground Server] VGMN model loaded (or retrieved from cache).")
        else:
            print("[Ground Server] Warning: VGMN model could not be loaded.")
        # --- 初始化其他组件 (未来实现) ---
        # ... (其余 __init__ 内容保持不变) ...

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
        msg_type = message.get('type')
        agent_id = message.get('agent_id')
        kf_id = message.get('kf_id')

        print(f"[Ground Server] Received message: Type={msg_type}, Agent={agent_id}, KF_ID={kf_id}")

        if msg_type == 'keyframe':
            if agent_id not in self.agent_keyframes:
                self.agent_keyframes[agent_id] = {}
                print(f"[Ground Server] Registered new agent: {agent_id}")

            kf_timestamp = message.get('timestamp')
            kf_local_pose = message.get('local_pose')
            kf_image_data = message.get('image') # 目前这个是 None

            if kf_id is not None:
                self.agent_keyframes[agent_id][kf_id] = {
                    'timestamp': kf_timestamp,
                    'local_pose': kf_local_pose,
                }
                print(f"[Ground Server] Stored keyframe {kf_id} for agent {agent_id}.")

                # --- 调用 VGMN 推理并存储结果 --- VVVV
                if self.vgmn_model:
                    print(f"[Ground Server] Running VGMN inference for Agent {agent_id}, KF {kf_id}...")
                    # kf_image_data 目前是 None，run_vgmn_inference 会处理这个（用假数据）
                    geo_pred_idx, features_np = run_vgmn_inference(self.vgmn_model, kf_image_data)

                    if features_np is not None:
                        if agent_id not in self.vgmn_results:
                            self.vgmn_results[agent_id] = {}
                        self.vgmn_results[agent_id][kf_id] = {
                            'geo_prediction_index': geo_pred_idx,
                            'features': features_np
                        }
                        print(f"[Ground Server] Stored VGMN results for Agent {agent_id}, KF {kf_id}. GeoPred: {geo_pred_idx}, FeatShape: {features_np.shape}")
                    else:
                        print(f"[Ground Server] VGMN inference failed for Agent {agent_id}, KF {kf_id}.")
                else:
                    print(f"[Ground Server] VGMN model not loaded, skipping inference for Agent {agent_id}, KF {kf_id}.")
                # --- VGMN 调用结束 ---

            else:
                 print(f"[Ground Server] Warning: Received keyframe message without kf_id from agent {agent_id}.")
        else:
            print(f"[Ground Server] Received unknown message type: {msg_type}")