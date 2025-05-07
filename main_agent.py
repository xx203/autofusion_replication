import time
import yaml
import os
import numpy as np
from uav_agent.agent import UAVAgent
from ground_server.server import GroundServer # <--- 导入 GroundServer 类

def load_config(config_path="configs/params.yaml"):
    """加载 YAML 配置文件"""
    print(f"Loading configuration from: {config_path}")
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return None
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("Configuration loaded successfully.")
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        return None

def run_simulation_with_server(config, duration_seconds=10): # <--- 修改函数，现在包含 Server
    """
    模拟运行多个 Agent，并将数据发送给一个共享的 Server 实例。

    Args:
        config (dict): 加载的配置。
        duration_seconds (int): 每个 Agent 模拟运行的总时长（秒）。
    """
    print("\n--- Initializing Ground Server ---")
    server = GroundServer(config=config) # <--- 创建 Server 实例

    num_agents_to_simulate = config.get('simulation', {}).get('num_uavs', 1)
    print(f"\n--- Starting Simulation for {num_agents_to_simulate} Agents ---")
    print(f"--- Each agent will run for approx. {duration_seconds} seconds ---")

    agents = []
    for i in range(num_agents_to_simulate):
        # 创建 Agent 时传入 server 实例
        agent = UAVAgent(agent_id=i, config=config, server_instance=server) # <--- 传递 server
        agents.append(agent)

    start_time = time.time()
    end_time = start_time + duration_seconds

    # --- 简单的循环模拟，所有 Agent 近似并行执行一步 ---
    frame_counts = [0] * num_agents_to_simulate
    while time.time() < end_time:
        current_time = time.time()
        for i, agent in enumerate(agents):
             # 模拟处理一帧数据
             agent.run_step(image=None, timestamp=current_time)
             frame_counts[i] += 1

        # 稍微暂停一下，模拟帧率
        time.sleep(0.05) # 所有 agent 跑完一轮后暂停

    total_duration = time.time() - start_time
    print(f"\n--- Simulation Finished ---")
    print(f"Total duration: {total_duration:.2f} seconds")
    for i, agent in enumerate(agents):
         print(f"  Agent {i}: Frames processed={frame_counts[i]}, Keyframes generated={agent.local_slam.keyframe_count}")

    # 在这里可以访问 Server 存储的数据 (虽然现在 Server 还没处理)
# --- 修改打印 VGMN 结果摘要 --- VVVV
    print("\n--- Server VGMN Results Summary ---")
    total_vgmn_results_count = 0 # 重命名变量以区分
    if server.vgmn_results:
        print(f"Total agents with VGMN results: {len(server.vgmn_results)}")
        for agent_id, agent_vgmn_data in server.vgmn_results.items(): # 重命名变量
            num_results_for_agent = len(agent_vgmn_data)
            total_vgmn_results_count += num_results_for_agent

            first_kf_id_processed = None
            first_result_details = "No results to show."

            if num_results_for_agent > 0:
                # 获取这个 agent 的第一个被处理的关键帧 ID
                first_kf_id_processed = list(agent_vgmn_data.keys())[0]
                if first_kf_id_processed in agent_vgmn_data:
                    result_entry = agent_vgmn_data[first_kf_id_processed]
                    geo_pred = result_entry.get('geo_prediction_index', 'N/A')
                    # 确保 features 是 numpy array 才调用 .shape
                    features_data = result_entry.get('features')
                    feat_shape = features_data.shape if isinstance(features_data, np.ndarray) else 'N/A'
                    first_result_details = f"KF {first_kf_id_processed} -> GeoPred: {geo_pred}, FeatShape: {feat_shape}"

            print(f"  Agent {agent_id} has {num_results_for_agent} VGMN results. (e.g., {first_result_details})")
        print(f"Total VGMN results processed by server: {total_vgmn_results_count}")
    else:
        print("No VGMN results were processed by the server.")
    # --- 修改结束 ---

    # ... ("== Simulation Completed ==" 保持不变) ...


    # --- 新增打印 VGMN 结果摘要 --- VVVV
    print("\n--- Server VGMN Results Summary ---")
    total_vgmn_results = 0
    if server.vgmn_results:
        print(f"Total agents with VGMN results: {len(server.vgmn_results)}")
        for agent_id, results in server.vgmn_results.items():
            num_results = len(results)
            total_vgmn_results += num_results
            # 只打印第一个结果的摘要作为示例
            first_kf_id = list(results.keys())[0] if num_results > 0 else None
            first_result_summary = "N/A"
            if first_kf_id:
                first_geo_pred = results[first_kf_id].get('geo_prediction_index', 'N/A')
                first_feat_shape = results[first_kf_id].get('features', np.array([])).shape
                first_result_summary = f"KF {first_kf_id} -> GeoPred: {first_geo_pred}, FeatShape: {first_feat_shape}"

            print(f"  Agent {agent_id} has {num_results} VGMN results. (e.g., {first_result_summary})")
        print(f"Total VGMN results processed by server: {total_vgmn_results}")
    else:
        print("No VGMN results were processed by the server.")
    # --- 新增结束 ---


# ... (脚本结束的打印信息 "== Simulation Completed ==" 保持不变) ...


# --- 脚本主入口 ---
if __name__ == "__main__":
    print("======================================")
    print("== Running Multi-UAV Simulation with Server ==")
    print("======================================")

    # 1. 加载配置
    config = load_config()

    if config:
        # 2. 运行包含 Server 的模拟
        simulation_duration = 15 # 秒，可以根据需要调整
        run_simulation_with_server(config=config,
                                   duration_seconds=simulation_duration)

        print("\n======================================")
        print("== Simulation Completed ==")
        print("======================================")
    else:
        print("Exiting due to configuration loading error.")