import yaml
import os
from ground_server.server import GroundServer # 从 ground_server 包导入我们刚写的 Server 类

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

# --- 脚本主入口 ---
if __name__ == "__main__":
    print("======================================")
    print("== Starting AutoFusion Ground Server ==")
    print("======================================")

    # 1. 加载配置
    config = load_config()

    if config:
        # 2. 创建 Ground Server 实例
        server = GroundServer(config=config)

        # 3. 启动服务器主循环
        # (这个循环会一直运行，直到你手动停止，例如按 Ctrl+C)
        server.run()

        print("\n======================================")
        print("== Ground Server Shut Down ==")
        print("======================================")
    else:
        print("Exiting due to configuration loading error.")