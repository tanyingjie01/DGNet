"""
GKSNets AI增强组件模块
包含基础的神经网络模块，以及方法中的修正模块（位置A、B、C）和数据路径的残差求解器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
import math

# 使用PyTorch Geometric的优化组件
from torch_geometric.nn import MLP, MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree


class MPNNLayer(MessagePassing):
    """
    消息传递神经网络单层，使用PyG的MessagePassing基类实现

    对点和边的特征进行编码，并使用消息传递机制进行聚合后，更新节点特征
    """
    
    def __init__(self, 
                 node_dim: int,
                 edge_dim: int,
                 hidden_dim: int,
                 aggregation: str = 'mean'):
        """
        初始化MPNN层
        
        Args:
            node_dim: 节点特征维度
            edge_dim: 边特征维度
            hidden_dim: MLP中隐藏层的维度
            aggregation: 聚合方式 ('mean', 'sum', 'max', 'add')
        """
        super().__init__(aggr=aggregation)
        
        self.node_feature_dim = node_dim # 节点特征维度，这里不能用self.node_dim，因为self.node_dim是PyG的属性，不能被修改
        self.edge_feature_dim = edge_dim # 边特征维度，这里不能用self.edge_dim，因为self.edge_dim是PyG的属性，不能被修改
        self.hidden_dim = hidden_dim
        
        # 消息函数，聚合一条边相关的所有信息
        self.message_mlp = MLP(
            in_channels=2 * node_dim + edge_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=2,
            act='relu'
        )
        
        # 更新函数，聚合所有节点信息，并更新特征
        self.update_mlp = MLP(
            in_channels=node_dim + hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=node_dim,
            num_layers=1,
            act='relu'
        )
    
    def forward(self, 
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        MPNN前向传播
        
        Args:
            node_features: [N, node_dim] 节点特征
            edge_index: [2, E] 边索引
            edge_attr: [E, edge_dim] 边特征
            
        Returns:
            torch.Tensor: [N, node_dim] 更新后的节点特征
        """
        # 如果边特征为空，则使用零向量
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.shape[1], self.edge_feature_dim, device=edge_index.device)
        
        # 使用PyG的消息传递机制
        return self.propagate(edge_index, x=node_features, edge_attr=edge_attr)
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """计算消息"""
        # 拼接源节点、目标节点和边特征
        message_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(message_input)
    
    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """更新节点特征"""
        # 拼接原始节点特征和聚合消息
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(update_input)


class MPNNProcessor(nn.Module):
    """多层MPNN处理器，可选是否带残差连接"""
    
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 hidden_dim: int,
                 num_layers: int = 3,
                 aggregation: str = 'mean',
                 residual: bool = True):
        """
        初始化多层MPNN处理器
        
        Args:
            node_dim: 节点特征维度
            edge_dim: 边特征维度
            hidden_dim: MLP的隐藏层维度
            num_layers: MPNN层数
            aggregation: 聚合方式
            residual: 是否使用残差连接
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.residual = residual
        
        # 多层MPNN - 使用ModuleList
        self.layers = nn.ModuleList([
            MPNNLayer(node_dim, edge_dim, hidden_dim, aggregation)
            for _ in range(num_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(node_dim) for _ in range(num_layers)
        ])
    
    def forward(self,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        多层MPNN前向传播
        
        Args:
            node_features: [N, node_dim] 节点特征
            edge_index: [2, E] 边索引
            edge_attr: [E, edge_dim] 边特征
            
        Returns:
            torch.Tensor: [N, node_dim] 处理后的节点特征
        """
        h = node_features
        
        # 消息传递更新特征
        for layer, norm in zip(self.layers, self.layer_norms):
            h_new = layer(h, edge_index, edge_attr)
            
            # 选择是否使用残差连接
            if self.residual:
                h = h + h_new
            else:
                h = h_new
            
            # 层归一化
            h = norm(h)
        
        return h

# 仅仅修正有边连接的算子
# 后续可以改成在数据预处理阶段，预先构建一个基于固定半径的、包含长程作用的图
# 注意，每个.py模块都手动double了边，后续可以尝试修改，比如PyG对这种情况有智能处理（将边特征写为节点特征之差）
class OperatorCorrector(nn.Module):
    """
    位置A：算子修正器
    
    学习对基础物理算子L的修正
    L_final = L_base + ΔL_learned
    """
    
    def __init__(self,
                 spatial_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 operator_dim: int = 1,
                 num_node_types: int = 3):
        """
        初始化算子修正器
        
        Args:
            spatial_dim: 空间维度（坐标维度）
            hidden_dim: 隐藏层维度
            num_layers: MPNN层数
            operator_dim: 输出的算子修正维度（L做为矩阵算子，每个元素是标量，所以输出维度为1）
            num_node_types: 节点类型数量（0=内部，1=dirichlet，2=neumann），默认为3
        """
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.operator_dim = operator_dim
        self.num_node_types = num_node_types
        
        # 节点编码器：输入为节点坐标 + 体积 + 节点类型one-hot
        self.node_encoder = MLP(
            in_channels=spatial_dim + 1 + num_node_types,  # 坐标 + 体积 + 节点类型one-hot
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=2,
            act='relu'
        )
        
        # 边编码器：输入为坐标差 + 标量距离
        self.edge_encoder = MLP(
            in_channels=spatial_dim + 1,  # 坐标差 + 距离
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=2,
            act='relu'
        )
        
        # MPNN处理器
        self.processor = MPNNProcessor(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        # 边修正解码器：输入为双节点特征，输出为修正值
        self.edge_corrector = MLP(
            in_channels=2 * hidden_dim, 
            hidden_channels=hidden_dim,
            out_channels=operator_dim,
            num_layers=2,
            act='relu'
        )
    
    def forward(self, 
                graph_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算算子修正ΔL，用于增强基础物理算子：L_final = L_base + ΔL_learned
        
        Args:
            graph_data: 包含图结构数据的字典，包含以下键值对：
                - 'nodes': torch.Tensor, shape [N, spatial_dim] 节点坐标
                - 'edges': torch.Tensor, shape [E, 2] 边索引（输入是无向边，即i_j边和j_i边不会重复出现，不能有重边，且不能包括自环边）
                - 'node_volumes': torch.Tensor, shape [N] 节点体积（几何信息）
                - 'node_type': torch.Tensor, shape [N] 节点类型（0=内部，1=dirichlet，2=neumann）
        
        Returns:
            torch.Tensor: shape [N, N] 算子修正矩阵 ΔL
        """
        
        # 提取原始数据
        nodes = graph_data['nodes']
        unique_edges = graph_data['edges'] # [E, 2]，我们称之为 unique_edges 以示区分
        node_volumes = graph_data.get('node_volumes', torch.ones(nodes.shape[0], device=nodes.device))
        node_type = graph_data.get('node_type', torch.zeros(nodes.shape[0], device=nodes.device))

        # 获取节点数量和设备
        N = nodes.shape[0]
        device = nodes.device
        
        # 将原始的 [E, 2] 边列表转换为 [2*E, 2] 的双向边列表
        edge_reversed = unique_edges.flip(dims=[1]) # 反转每条边的方向
        edges_bidirectional = torch.cat([unique_edges, edge_reversed], dim=0) # 拼接成双向边
        
        # 转置为PyG格式
        edge_index = edges_bidirectional.T  # [2, 2*E]

        # 计算边特征
        src_coords = nodes[edges_bidirectional[:, 0]] # [2*E, spatial_dim] 源节点坐标
        dst_coords = nodes[edges_bidirectional[:, 1]] # [2*E, spatial_dim] 目标节点坐标
        coord_diffs = dst_coords - src_coords  # [2*E, spatial_dim]，自然地包含了两个方向的坐标差
        distances = torch.norm(coord_diffs, dim=1, keepdim=True) # [2*E, 1] 距离
        
        # 边特征：坐标差 + 距离，编码后输出到隐藏层
        edge_features = torch.cat([coord_diffs, distances], dim=1)
        encoded_edges = self.edge_encoder(edge_features)  # [2*E, hidden_dim]

        # 节点特征：节点坐标 + 节点体积 + 节点类型one-hot，编码后输出到隐藏层
        node_type_onehot = F.one_hot(node_type.long(), num_classes=self.num_node_types).float()
        node_input = torch.cat([nodes, node_volumes.unsqueeze(-1), node_type_onehot], dim=-1)
        encoded_nodes = self.node_encoder(node_input)

        # MPNN处理，输出每个节点的特征
        processed_nodes = self.processor(encoded_nodes, edge_index, encoded_edges)
        src_nodes = processed_nodes[edges_bidirectional[:, 0]] # [2*E, hidden_dim] 源节点特征
        dst_nodes = processed_nodes[edges_bidirectional[:, 1]] # [2*E, hidden_dim] 目标节点特征
        
        # 解码器输入：只使用经过MPNN处理的源节点特征 + 目标节点特征
        edge_inputs = torch.cat([
            src_nodes,          # [2*E, hidden_dim] 源节点特征
            dst_nodes,          # [2*E, hidden_dim] 目标节点特征
        ], dim=1)
        
        # 解码器输出
        raw_corrections = self.edge_corrector(edge_inputs) # [2*E, 1]

        # 新增：对修正值进行缩放和限制，以确保数值稳定性
        # 物理扩散系数 alpha = k/(rho*c) 约为 1.4e-5
        # 我们使用一个同量级的缩放因子来限制AI修正的强度
        scaling_factor = 1e-5
        corrections = scaling_factor * torch.tanh(raw_corrections)

        # 构建修正矩阵
        delta_L = torch.zeros(N, N, device=device)
        
        # 使用双向边的索引直接填充矩阵
        # delta_L[i,j] 表示节点j对节点i的影响
        src, dst = edges_bidirectional[:, 0], edges_bidirectional[:, 1]
        delta_L[dst, src] = corrections.squeeze(-1) # 注意：PyG消息传递定义中，消息从j->i，所以src是j，dst是i。修正L_ij表示j对i的作用。
                                                     # 在我们的定义中，delta_L[i,j]表示j对i的影响，所以索引是 (i, j)
                                                     # 为了清晰，我们还是用 (dst, src) 对应 (i, j)
        
        # 填充对角线，保证行和为0
        row_sums = delta_L.sum(dim=1)
        delta_L[torch.arange(N, device=device), torch.arange(N, device=device)] = -row_sums
        
        return delta_L

# 修正系数仅针对有边连接的节点对
# 后续同样可以改成在数据预处理阶段，预先构建一个基于固定半径的、包含长程作用的图（与位置A用相同的图）
class AttentionCombiner(nn.Module):
    """
    位置C：注意力组合器
    
    学习时间-空间注意力权重，替代固定的时间卷积，实现智能的历史信息权重分配
    """
    
    def __init__(self,
                 spatial_dim: int,
                 feature_dim: int,
                 hidden_dim: int = 64,
                 num_heads: int = 4):
        """
        初始化注意力组合器
        
        Args:
            spatial_dim: 空间维度
            feature_dim: 特征维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
        """
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Query编码器，编码当前状态信息
        self.query_encoder = MLP(
            in_channels=feature_dim + spatial_dim + 1,  # 特征 + 坐标 + 时间
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=2,
            act='relu'
        )
        
        # Key编码器，编码历史状态信息（value和key相同）
        self.key_encoder = MLP(
            in_channels=feature_dim + spatial_dim + 1,  # 特征 + 坐标 + 时间
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=2,
            act='relu'
        )
        
        # 多头注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
    
    def forward(self,
                current_state: torch.Tensor,
                current_coords: torch.Tensor, 
                current_time: float,
                history_states: torch.Tensor,
                history_coords: torch.Tensor,
                history_times: torch.Tensor) -> torch.Tensor:
        """
        计算时间-空间注意力权重
        
        Args:
            current_state: [N, feature_dim] 当前状态
            current_coords: [N, spatial_dim] 当前坐标
            current_time: 当前时间
            history_states: [T, N, feature_dim] 历史状态序列
            history_coords: [N, spatial_dim] 坐标（假设不变）
            history_times: [T] 历史时间序列
            
        Returns:
            torch.Tensor: [N, T] 每个节点对每个历史时刻的注意力权重
        """
        # 获取节点数量和历史时刻数量
        N = current_state.shape[0]
        T = history_states.shape[0]
        device = current_state.device
        
        # 编码 Query (当前状态)
        # 将当前时间扩展，使其能与每个节点的状态和坐标拼接
        current_time_tensor = torch.full((N, 1), current_time, device=device)
        # 拼接成Query的输入: [特征, 坐标, 时间]
        query_input = torch.cat([current_state, current_coords, current_time_tensor], dim=1)
        # 将所有节点的当前状态编码成Query向量
        queries = self.query_encoder(query_input)  # 输出形状: [N, hidden_dim]
        
        # 编码 Key 和 Value (历史状态)
        # 扩展历史时间张量以匹配历史状态的形状
        history_times_expanded = history_times.view(T, 1, 1).expand(T, N, 1) # 形状: [T, N, 1]
        # 扩展坐标张量以匹配历史状态的形状
        history_coords_expanded = history_coords.unsqueeze(0).expand(T, N, -1) # 形状: [T, N, spatial_dim]
        
        # 拼接成Key的输入: [历史特征, 历史坐标, 历史时间]
        key_inputs = torch.cat([
            history_states,
            history_coords_expanded,
            history_times_expanded
        ], dim=2) # 形状: [T, N, D_in]，这里D_in是特征维度 + 空间维度 + 时间维度
        
        # 为了送入MLP，先重塑成一个大的批次，即压平，形状为[T*N, D_in]
        key_inputs = key_inputs.reshape(T * N, -1)
        # 编码所有历史状态，得到Key向量，输出形状: [T*N, hidden_dim]
        keys = self.key_encoder(key_inputs)
        # 再重塑回时间序列的形状，输出形状: [T, N, hidden_dim]
        keys = keys.reshape(T, N, self.hidden_dim)
        
        # 在标准的注意力机制中，Value通常和Key是同一个来源
        values = keys
        
        # 准备注意力模块的输入
        # nn.MultiheadAttention期望的输入形状是 [batch_size, sequence_length, feature_dim]
        
        # 将queries处理成批次大小为N，序列长度为1的形状
        queries = queries.unsqueeze(1)  # 形状: [N, 1, hidden_dim]
        
        # 将keys和values的维度进行转置，使其批次大小为N，序列长度为T
        keys = keys.permute(1, 0, 2)    # 形状: [N, T, hidden_dim]
        values = values.permute(1, 0, 2)  # 形状: [N, T, hidden_dim]
        
        # 执行注意力计算，并直接获取权重
        # 我们只关心第二个返回值 attention_weights，它就是我们需要的权重矩阵
        # 第一个返回值 _ 是加权求和后的上下文向量，这里我们用不上
        _ , attention_weights = self.attention(
            query=queries, 
            key=keys, 
            value=values,
            need_weights=True,     # 关键：告诉模块，我们需要返回权重
            average_attn_weights=True # 关键：将多头的权重平均，简化输出
        )
        # `attention_weights` 的输出形状是 [N, 1, T]
        
        # 整理形状并返回
        # 去掉中间多余的维度，得到最终的 [N, T] 权重矩阵
        final_weights = attention_weights.squeeze(1)
        
        return final_weights


class NonlinearDynamicsSolver(nn.Module):
    """
    新增模块：非线性动力学求解器 r(u^k) ≈ N(u^k)
    
    学习物理方程中的非线性项 N(u)
    采用与ResidualSolver类似的"编码-处理-解码"架构
    """
    
    def __init__(self,
                 spatial_dim: int,
                 node_feature_dim: int,
                 output_dim: int,
                 hidden_dim: int = 128,
                 num_processing_layers: int = 5,
                 edge_dim: int = 8,
                 num_node_types: int = 3):
        """
        初始化非线性动力学求解器
        
        Args:
            spatial_dim: 空间维度
            node_feature_dim: 节点特征维度
            output_dim: 输出维度 (与特征维度相同)
            hidden_dim: 隐藏层维度
            num_processing_layers: MPNN堆叠的GNN层数
            edge_dim: 边特征维度
            num_node_types: 节点类型数量
        """
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.output_dim = output_dim
        self.num_node_types = num_node_types
        
        # 节点编码器
        self.node_encoder = MLP(
            in_channels=node_feature_dim + spatial_dim + num_node_types,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=2,
            act='relu'
        )
        
        # 边编码器
        self.edge_encoder = MLP(
            in_channels=spatial_dim + node_feature_dim + 1,
            hidden_channels=hidden_dim,
            out_channels=edge_dim,
            num_layers=1,
            act='relu'
        )
        
        # 处理器
        self.processor = MPNNProcessor(
            node_dim=hidden_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_processing_layers,
            residual=True
        )
        
        # 解码器
        self.decoder = MLP(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            num_layers=3,
            act='relu'
        )

    def forward(self, graph_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算非线性项 r(u^k)
        
        Args:
            graph_data: 包含图结构和物理量的字典
            
        Returns:
            torch.Tensor: [N, output_dim] 非线性项 r(u^k)
        """
        nodes = graph_data['nodes']
        unique_edges = graph_data['edges']
        node_features = graph_data['node_features']
        node_type = graph_data.get('node_type', torch.zeros(nodes.shape[0], device=nodes.device))
        
        # 将无向边转换为双向边
        edge_reversed = unique_edges.flip(dims=[1])
        edges_bidirectional = torch.cat([unique_edges, edge_reversed], dim=0)
        edge_index = edges_bidirectional.T
        
        # 节点编码
        node_type_onehot = F.one_hot(node_type.long(), num_classes=self.num_node_types).float()
        node_input = torch.cat([node_features, nodes, node_type_onehot], dim=1)
        encoded_nodes = self.node_encoder(node_input)
        
        # 边编码
        src_coords = nodes[edges_bidirectional[:, 0]]
        dst_coords = nodes[edges_bidirectional[:, 1]]
        src_features = node_features[edges_bidirectional[:, 0]]
        dst_features = node_features[edges_bidirectional[:, 1]]
        coord_diffs = dst_coords - src_coords
        feature_diffs = dst_features - src_features
        distances = torch.norm(coord_diffs, dim=1, keepdim=True)
        edge_features = torch.cat([coord_diffs, feature_diffs, distances], dim=1)
        encoded_edges = self.edge_encoder(edge_features)
        
        # GNN处理
        processed_nodes = self.processor(encoded_nodes, edge_index, encoded_edges)
        
        # 解码
        r_uk = self.decoder(processed_nodes)
        
        return r_uk

class ResidualSolver(nn.Module):
    """
    数据路径：残差求解器
    
    学习物理路径无法捕获的复杂效应，采用"编码-处理-解码"架构
    """
    
    def __init__(self,
                 spatial_dim: int,
                 node_feature_dim: int,
                 output_dim: int,
                 hidden_dim: int = 128,
                 num_processing_layers: int = 5,
                 edge_dim: int = 8,
                 num_node_types: int = 3):
        """
        初始化残差求解器
        
        Args:
            spatial_dim: 空间维度
            node_feature_dim: 节点特征维度
            output_dim: 输出维度
            hidden_dim: 隐藏层维度
            num_processing_layers: MPNN堆叠的GNN层数
            edge_dim: 边特征维度
            num_node_types: 节点类型数量（0=内部，1=dirichlet，2=neumann），默认为3
        """
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.output_dim = output_dim
        self.num_node_types = num_node_types
        
        # 节点编码器：传入物理特征 + 坐标 + 节点类型one-hot
        self.node_encoder = MLP(
            in_channels=node_feature_dim + spatial_dim + num_node_types,  # 特征 + 坐标 + 节点类型one-hot
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=2,
            act='relu'
        )
        
        # 边编码器：传入坐标差 + 物理量差 + 距离
        self.edge_encoder = MLP(
            in_channels=spatial_dim + node_feature_dim + 1,  # 坐标差 + 物理量差 + 距离
            hidden_channels=hidden_dim,
            out_channels=edge_dim,
            num_layers=1,
            act='relu'
        )
        
        # 处理器：多层MPNN进行图推理
        self.processor = MPNNProcessor(
            node_dim=hidden_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_processing_layers,
            residual=True
        )
        
        # 解码器：使用PyG的MLP
        self.decoder = MLP(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            num_layers=3,
            act='relu'
        )
    
    def forward(self, graph_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算残差解u_net，用于学习物理模型无法捕获的复杂效应
        
        Args:
            graph_data: 包含图结构数据和物理量的字典，包含以下键值对：
                - 'nodes': torch.Tensor, shape [N, spatial_dim] 节点坐标        
                - 'edges': torch.Tensor, shape [E, 2] 边索引（输入是无向边，即i_j边和j_i边不会重复出现，不能有重边，且不能包括自环边）
                - 'node_features': torch.Tensor, shape [N, node_feature_dim] 每个节点的物理特征
                - 'boundary_info': Dict[str, Any], 包含边界条件信息的字典
                - 'node_type': torch.Tensor, shape [N] 节点类型（0=内部，1=dirichlet，2=neumann）（可选输入，默认为全零）
            
        Returns:
            torch.Tensor: shape [N, output_dim] 残差解 u_net，用于增强物理求解器的结果：u_final = u_physics + u_net
        """

        # 获取图结构数据
        nodes = graph_data['nodes']  # [N, spatial_dim]
        unique_edges = graph_data['edges']  # [E, 2] 
        node_features = graph_data['node_features']  # [N, feature_dim]
        boundary_info = graph_data.get('boundary_info', {}) # 获取边界信息
        node_type = graph_data.get('node_type', torch.zeros(nodes.shape[0], device=nodes.device))
        
        # 获取节点数量和设备
        N = nodes.shape[0]
        device = nodes.device
        
        # 将原始的 [E, 2] 边列表转换为 [2*E, 2] 的双向边列表
        edge_reversed = unique_edges.flip(dims=[1])  # 反转每条边的方向
        edges_bidirectional = torch.cat([unique_edges, edge_reversed], dim=0)  # 拼接成双向边
        
        # 转置为PyG格式
        edge_index = edges_bidirectional.T  # [2, 2*E]
        
        # 节点特征：物理特征 + 坐标 + 节点类型one-hot
        node_type_onehot = F.one_hot(node_type.long(), num_classes=self.num_node_types).float()
        node_input = torch.cat([node_features, nodes, node_type_onehot], dim=1)

        # 编码节点特征
        encoded_nodes = self.node_encoder(node_input)  # [N, hidden_dim]编码后的节点特征
        h_anchor = encoded_nodes.clone() # 保存初始编码用于特征锚定
        
        # 计算边特征：坐标差、物理量差和距离
        src_coords = nodes[edges_bidirectional[:, 0]]  # [2*E, spatial_dim] 源节点坐标
        dst_coords = nodes[edges_bidirectional[:, 1]]  # [2*E, spatial_dim] 目标节点坐标
        src_features = node_features[edges_bidirectional[:, 0]]  # [2*E, feature_dim] 源节点物理特征
        dst_features = node_features[edges_bidirectional[:, 1]]  # [2*E, feature_dim] 目标节点物理特征
        coord_diffs = dst_coords - src_coords  # [2*E, spatial_dim] 坐标差，自然地包含了两个方向的坐标差
        feature_diffs = dst_features - src_features  # [2*E, feature_dim] 物理量差
        distances = torch.norm(coord_diffs, dim=1, keepdim=True)  # [2*E, 1] 距离
        
        # 拼接边特征：[坐标差, 物理量差, 距离]
        edge_features = torch.cat([coord_diffs, feature_diffs, distances], dim=1)  # [2*E, spatial_dim + feature_dim + 1]

        # 编码边特征
        encoded_edges = self.edge_encoder(edge_features)  # [2*E, edge_dim]编码后的边特征
        
        # --- 手动执行GNN处理循环，以插入特征锚定 ---
        h_current = encoded_nodes
        dirichlet_indices = boundary_info.get('dirichlet', {}).get('indices')

        for i in range(self.processor.num_layers):
            # a. 调用单个MPNN层
            h_updated = self.processor.layers[i](h_current, edge_index, encoded_edges)
            
            # b. 应用残差连接和层归一化
            if self.processor.residual:
                h_current = h_current + h_updated
            else:
                h_current = h_updated
            h_current = self.processor.layer_norms[i](h_current)
            
            # c. 特征锚定：重置狄利克雷边界节点的隐藏状态
            if dirichlet_indices is not None:
                h_current[dirichlet_indices] = h_anchor[dirichlet_indices]
        
        processed_nodes = h_current
        
        # 解码输出
        u_net = self.decoder(processed_nodes)
        
        return u_net
