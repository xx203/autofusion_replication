import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models # 导入预训练的 CNN 模型

# 尝试导入 PyG 的 GCNConv 层
try:
    from torch_geometric.nn import GCNConv, global_mean_pool # global_mean_pool 用于图级别特征聚合
    pyg_available = True
except ImportError:
    print("Warning: PyTorch Geometric or GCNConv not found. VGMN will not use GCN layers.")
    # 定义一个假的 GCNConv 以免报错，但它什么也不做
    class GCNConv:
        def __init__(self, in_channels, out_channels):
            print("Using Dummy GCNConv (PyG not fully working)")
            self.dummy_layer = nn.Identity() # 一个什么也不做的层
        def __call__(self, x, edge_index):
            # 忽略 edge_index，直接返回输入
            return self.dummy_layer(x)
    # 定义一个假的 global_mean_pool
    def global_mean_pool(x, batch):
        # 简单的对节点特征求平均，忽略 batch 信息
        return torch.mean(x, dim=0, keepdim=True) # 或者 dim=1? 取决于输入维度
    pyg_available = False


class VGMN(nn.Module):
    """
    VGMN 模型的基础框架。
    包含 CNN Backbone 和 GCN (如果 PyG 可用)。
    """
    def __init__(self, config):
        """
        初始化 VGMN 模型。

        Args:
            config (dict): 包含模型参数的配置字典，例如：
                           config['vgmn']['cnn_backbone'] = 'resnet18'
                           config['vgmn']['gcn_hidden_dim'] = 512
                           config['vgmn']['feature_dim'] = 256 # VGMN 输出的最终特征维度
                           config['vgmn']['num_geo_classes'] = 1000 # 假设的地理位置分类数量
        """
        super().__init__()
        self.config = config
        self.pyg_available = pyg_available
        print("[VGMN Model] Initializing...")

        # --- 1. CNN Backbone ---
        cnn_backbone_name = self.config.get('vgmn', {}).get('cnn_backbone', 'resnet18')
        print(f"[VGMN Model] Using CNN Backbone: {cnn_backbone_name}")
        if cnn_backbone_name == 'resnet18':
            # 加载预训练的 ResNet-18
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            # 去掉最后的平均池化层和全连接层
            self.cnn_feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
            cnn_output_dim = 512 # ResNet-18 block4 输出的通道数
        elif cnn_backbone_name == 'resnet34':
             resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
             self.cnn_feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
             cnn_output_dim = 512 # ResNet-34 block4 输出通道数
        # 可以添加对其他 backbone 的支持 (VGG, etc.)
        else:
            raise ValueError(f"Unsupported CNN backbone: {cnn_backbone_name}")

        # --- 2. GCN Layers (如果 PyG 可用) ---
        if self.pyg_available:
            gcn_hidden_dim = self.config.get('vgmn', {}).get('gcn_hidden_dim', 512)
            # GCN 输入维度是 CNN 输出的通道数
            # GCN 输出维度也是隐藏层维度（可以多层）
            self.gcn1 = GCNConv(cnn_output_dim, gcn_hidden_dim)
            self.gcn2 = GCNConv(gcn_hidden_dim, gcn_hidden_dim) # 可以再加一层 GCN
            gcn_output_dim = gcn_hidden_dim
            print(f"[VGMN Model] GCN Layers enabled: Input={cnn_output_dim}, Hidden={gcn_hidden_dim}")
        else:
            # 如果 PyG 不可用或 GCN 未加载，GCN 输出维度等于 CNN 输出维度
            gcn_output_dim = cnn_output_dim
            print("[VGMN Model] GCN Layers disabled or PyG not fully functional.")


        # --- 3. 输出层 ---
        # 用于生成最终的匹配/检索特征的线性层
        final_feature_dim = self.config.get('vgmn', {}).get('feature_dim', 256)
        self.feature_projection = nn.Linear(gcn_output_dim, final_feature_dim)

        # (可选) 用于地理位置分类的线性层
        num_geo_classes = self.config.get('vgmn', {}).get('num_geo_classes', 1000) # 需要根据你的数据集确定
        self.geolocation_classifier = nn.Linear(final_feature_dim, num_geo_classes)

        print(f"[VGMN Model] Output Feature Dim: {final_feature_dim}")
        print(f"[VGMN Model] Geolocation Classes: {num_geo_classes}")
        print("[VGMN Model] Initialization complete.")


    def forward(self, image_batch, edge_index=None, batch_map=None):
        """
        VGMN 的前向传播。

        Args:
            image_batch (torch.Tensor): 输入的图像批次，形状如 (B, C, H, W)。
            edge_index (torch.Tensor, optional): 图的边索引，形状如 (2, NumEdges)。
                                                 只有在 PyG 可用时才需要。
            batch_map (torch.Tensor, optional): 将每个节点映射到其所属批次索引的张量，
                                                形状如 (NumNodes)。用于图池化。

        Returns:
            tuple: (geo_logits, final_features)
                   geo_logits (torch.Tensor): 地理位置分类的 logits，形状 (B, num_geo_classes)。
                   final_features (torch.Tensor): 用于匹配/检索的特征，形状 (B, final_feature_dim)。
        """
        # 1. 通过 CNN 提取特征图
        # 输入: (B, C, H, W)
        # 输出: (B, C_cnn, H_out, W_out)
        cnn_features = self.cnn_feature_extractor(image_batch)
        # print("CNN Output shape:", cnn_features.shape)

        # --- 准备 GCN 输入 ---
        # 需要将 CNN 的输出特征图转换成图节点表示
        # 这是一个关键步骤，论文中可能没有详述，需要我们设计
        # 简单方法：将每个像素或每个 patch 作为一个节点
        # 复杂方法：使用 superpixel 或其他方法构建图

        # 假设我们将每个像素 (在 H_out, W_out 维度上) 作为一个节点
        B, C_cnn, H_out, W_out = cnn_features.shape
        num_nodes_per_image = H_out * W_out
        # 将特征图展平成节点特征：(B, C_cnn, H_out * W_out) -> (B * H_out * W_out, C_cnn)
        # 需要小心处理 batch 维度
        # 这是一个简化的假设，实际实现可能更复杂，需要配合 edge_index 和 batch_map
        # x = cnn_features.permute(0, 2, 3, 1).reshape(-1, C_cnn) # (B*H_out*W_out, C_cnn)
        
        # --- 在这个框架的初级阶段，我们先跳过复杂的图构建和 GCN 处理 ---
        # --- 我们直接在 CNN 特征图上进行池化 ---

        # 2. (简化) 全局平均池化 CNN 特征
        # 输入: (B, C_cnn, H_out, W_out)
        # 输出: (B, C_cnn)
        pooled_features = F.adaptive_avg_pool2d(cnn_features, (1, 1)).squeeze(-1).squeeze(-1)
        # print("Pooled Features shape:", pooled_features.shape)

        # --- 如果 PyG 可用且图构建完成，这里应该是 GCN 处理和图池化 ---
        # if self.pyg_available and edge_index is not None and batch_map is not None:
        #     x = self.gcn1(x, edge_index)
        #     x = F.relu(x)
        #     x = self.gcn2(x, edge_index)
        #     x = F.relu(x)
        #     # 使用图池化将节点特征聚合为图级别特征
        #     pooled_features = global_mean_pool(x, batch_map) # (B, gcn_hidden_dim)
        #     print("[VGMN Model] Using GCN path.")
        # else:
        #     # Fallback 或 GCN 未启用，使用上面池化的 CNN 特征
        #      print("[VGMN Model] Using CNN pooling path.")
        #      pass # pooled_features 已经是 CNN 池化结果了

        # 3. 特征投影
        # 输入: (B, C_cnn or gcn_output_dim)
        # 输出: (B, final_feature_dim)
        final_features = self.feature_projection(pooled_features)
        final_features = F.relu(final_features) # 可以加个激活函数
        # print("Final Features shape:", final_features.shape)

        # 4. (可选) 地理位置分类
        # 输入: (B, final_feature_dim)
        # 输出: (B, num_geo_classes)
        geo_logits = self.geolocation_classifier(final_features)
        # print("Geo Logits shape:", geo_logits.shape)

        return geo_logits, final_features

# --- 用于测试模型结构的代码 ---
if __name__ == '__main__':
    # 模拟配置
    test_config = {
        'vgmn': {
            'cnn_backbone': 'resnet18',
            'gcn_hidden_dim': 512,
            'feature_dim': 256,
            'num_geo_classes': 1000
        }
    }

    # 创建模型
    model = VGMN(config=test_config)
    # print(model) # 打印模型结构

    # 创建模拟输入数据
    # 假设 Batch Size = 4, 图像为 3 通道，224x224 像素
    dummy_image_batch = torch.randn(4, 3, 224, 224)

    # --- 模拟图结构数据 (如果需要测试 GCN 路径) ---
    # 假设每个图像的 CNN 输出是 (512, 7, 7)，即 49 个节点
    num_nodes_per_image = 7 * 7
    total_nodes = 4 * num_nodes_per_image
    # 创建假的边索引 (例如，每个节点连接自己)
    dummy_edge_index = torch.arange(total_nodes).unsqueeze(0).repeat(2, 1)
    # 创建假的 batch 映射
    dummy_batch_map = torch.repeat_interleave(torch.arange(4), num_nodes_per_image)

    # 前向传播测试
    print("\nTesting forward pass...")
    # 注意：在当前简化版中，我们不传入 edge_index 和 batch_map
    # geo_logits, final_features = model(dummy_image_batch, dummy_edge_index, dummy_batch_map)
    geo_logits, final_features = model(dummy_image_batch) # 调用简化版 forward


    print("Output Geo Logits shape:", geo_logits.shape)
    print("Output Final Features shape:", final_features.shape)