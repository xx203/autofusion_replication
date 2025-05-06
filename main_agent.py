import time
import yaml
import os
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
    print("\n--- Server Keyframe Log Summary ---")
    total_kfs_in_server = 0
    if server.agent_keyframes: # 检查字典是否为空
         print(f"Total agents registered in server: {len(server.agent_keyframes)}")
         for agent_id, kfs in server.agent_keyframes.items():
             num_kfs = len(kfs)
             total_kfs_in_server += num_kfs
             print(f"  Agent {agent_id} sent {num_kfs} keyframes (logged by server).")
         print(f"Total keyframes logged by server: {total_kfs_in_server}")
    else:
         print("No keyframes were logged by the server.")
    print(f"Total agents registered in server: {len(server.agent_keyframes)}")
    # for agent_id, kfs in server.agent_keyframes.items(): # <--- Server 现在还没存储数据
    #     print(f"  Agent {agent_id} sent {len(kfs)} keyframes (according to server log).")


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