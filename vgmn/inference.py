import torch
import yaml
import os
import numpy as np
# 从同级目录导入模型定义
try:
    from .models import VGMN
except ImportError:
    # 如果直接运行 inference.py，可能需要不同的导入方式
    from models import VGMN

# 图像预处理（需要 torchvision）
try:
    import torchvision.transforms as transforms
    from PIL import Image # 用于打开图像文件 (未来可能需要)
    torchvision_available = True
except ImportError:
    print("Warning: torchvision or PIL not found. Image preprocessing will be basic.")
    torchvision_available = False

# 全局变量来缓存加载的模型，避免重复加载
loaded_model = None
model_config = None

def load_vgmn_model(config):
    """
    加载 VGMN 模型。
    如果模型已经加载，则直接返回缓存的模型。

    Args:
        config (dict): 包含模型配置和权重的路径 (虽然现在没用到权重)。

    Returns:
        VGMN: 加载的模型实例。
    """
    global loaded_model, model_config
    if loaded_model is not None:
        print("[VGMN Inference] Returning cached model.")
        return loaded_model

    print("[VGMN Inference] Loading VGMN model...")
    model_config = config # 保存配置供后续使用
    model = VGMN(config=config)

    # --- 未来加载训练好的权重 ---
    # model_path = config.get('vgmn', {}).get('model_path', None)
    # if model_path and os.path.exists(model_path):
    #     try:
    #         model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # 加载到 CPU
    #         print(f"[VGMN Inference] Loaded weights from {model_path}")
    #     except Exception as e:
    #         print(f"[VGMN Inference] Error loading weights from {model_path}: {e}. Using untrained model.")
    # else:
    #     print("[VGMN Inference] Model path not specified or not found. Using untrained model.")
    print("[VGMN Inference] Using untrained model (no weights loaded).") # 当前状态

    model.eval() # !! 非常重要：将模型设置为评估模式 (禁用 dropout, batchnorm 使用运行统计数据)
    loaded_model = model # 缓存模型
    print("[VGMN Inference] Model loaded and set to evaluation mode.")
    return loaded_model

def preprocess_image(image_data):
    """
    对输入图像进行预处理，使其符合模型输入要求。

    Args:
        image_data: 图像数据。可以是 PIL Image 对象，或者 numpy 数组，
                     或者在这个模拟阶段可以是 None。

    Returns:
        torch.Tensor or None: 预处理后的图像张量 (需要添加 batch 维度)，或者 None。
    """
    if image_data is None:
        # --- 模拟阶段：如果没给图像，就创建一个假的输入 ---
        print("[VGMN Inference Preprocessing] No image data provided, creating dummy tensor.")
        # 返回一个符合模型输入的假张量 (例如 B=1, C=3, H=224, W=224)
        return torch.randn(1, 3, 224, 224)

    if not torchvision_available:
        print("Warning: torchvision not available, cannot perform standard preprocessing.")
        # 只能做非常基础的转换，或者直接报错/返回 None
        return None

    # --- 标准图像预处理流程 (需要 torchvision 和 PIL) ---
    # 1. 定义预处理变换 (通常与训练时使用的变换一致)
    #    这里的均值和标准差是 ImageNet 常用的值
    preprocess = transforms.Compose([
        transforms.Resize(256),             # 调整大小
        transforms.CenterCrop(224),         # 中心裁剪
        transforms.ToTensor(),              # 转为 Tensor (范围 [0, 1])
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 标准化
    ])

    # 2. 应用变换
    try:
        # 假设输入是 PIL Image
        # 如果输入是 numpy array (H, W, C)，需要先转成 PIL Image: image = Image.fromarray(image_data)
        input_tensor = preprocess(image_data)
        # 添加 Batch 维度 (B=1)
        input_batch = input_tensor.unsqueeze(0)
        return input_batch
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

def run_vgmn_inference(model, image_data):
    """
    使用加载的模型对预处理后的图像进行推理。

    Args:
        model (VGMN): 加载的 VGMN 模型。
        image_data: 原始图像数据 (或 None)。

    Returns:
        tuple or None: (geo_prediction, features)
                       geo_prediction (int or None): 预测的地理类别索引 (最可能的类别)。
                       features (np.array or None): 提取到的特征向量 (numpy 数组)。
                       如果出错则返回 None, None。
    """
    # 1. 预处理图像
    input_batch = preprocess_image(image_data)
    if input_batch is None:
        print("[VGMN Inference] Preprocessing failed or no data.")
        return None, None

    # 2. 模型推理 (不需要计算梯度)
    with torch.no_grad(): # 关闭梯度计算，节省内存和计算量
        try:
            # 注意：当前模型的 forward 不需要 edge_index 和 batch_map
            geo_logits, final_features = model(input_batch)
        except Exception as e:
            print(f"Error during model inference: {e}")
            return None, None

    # 3. 处理输出
    # 获取最终特征 (B=1, feature_dim) -> (feature_dim)
    features_np = final_features.squeeze(0).cpu().numpy() # 转为 numpy 数组

    # 获取地理位置预测 (B=1, num_classes) -> (预测的类别索引)
    # 这里我们简单地取概率最高的类别作为预测
    geo_prediction_index = torch.argmax(geo_logits, dim=1).item()

    # --- 未来可能需要将类别索引映射回真实的地理瓦片或坐标 ---
    # geo_info = index_to_geo_tile(geo_prediction_index, model_config)

    # print(f"[VGMN Inference] Predicted Geo Class: {geo_prediction_index}")
    # print(f"[VGMN Inference] Extracted Features shape: {features_np.shape}")

    return geo_prediction_index, features_np


# --- 用于测试推理接口的代码 ---
if __name__ == '__main__':
    print("--- Testing VGMN Inference ---")
    # 1. 加载模拟配置 (实际应该从文件加载)
    test_config = {
        'vgmn': {
            'cnn_backbone': 'resnet18',
            'gcn_hidden_dim': 512,
            'feature_dim': 256,
            'num_geo_classes': 1000,
            'model_path': None # 不加载权重
        }
    }

    # 2. 加载模型 (只会加载一次)
    model = load_vgmn_model(test_config)
    model_again = load_vgmn_model(test_config) # 第二次调用，应该会用缓存

    if model:
        # 3. 运行推理 (使用 None 作为图像数据，会生成假的输入)
        print("\nRunning inference with dummy data...")
        geo_pred, features = run_vgmn_inference(model, image_data=None)

        if features is not None:
            print(f"\nInference Successful:")
            print(f"  Predicted Geo Class Index: {geo_pred}")
            print(f"  Extracted Features Shape: {features.shape}")
            # print(f"  Features (first 5): {features[:5]}") # 打印部分特征值看看
        else:
            print("\nInference Failed.")
    else:
        print("Model loading failed.")