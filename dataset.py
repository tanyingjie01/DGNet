"""
GKSNets数据管道模块
包含PDE轨迹数据的结构定义、数据加载、预处理和批处理功能
"""

import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod


class DGGraph:
    """
    GKSNets的核心数据结构类，封装单个PDE训练轨迹的完整信息
    
    包含：
    - 几何信息：节点坐标、边连接关系、面连接关系
    - 时间序列：多个时间步的物理量演化
    - 源项信息：每个时间步的源项f(x,t)
    - 初始条件：u0(x)
    - 边界条件：支持Dirichlet和Neumann边界条件
    """
    
    def __init__(self, 
                 nodes: torch.Tensor,           # [N, spatial_dim] 节点坐标
                 edges: torch.Tensor,           # [E, 2] 边连接关系
                 faces: torch.Tensor,           # [F, 3] 面连接关系
                 node_features: torch.Tensor,   # [T, N, feature_dim] 时间序列物理量
                 source_terms: torch.Tensor,    # [T, N, source_dim] 源项序列
                 initial_condition: torch.Tensor, # [N, state_dim] 初始条件
                 time_points: torch.Tensor,     # [T] 时间点
                 edge_attr: Optional[torch.Tensor] = None, # [E, edge_dim] 边属性
                 boundary_info: Optional[Dict[str, Any]] = None,  # 边界条件信息
                 node_type: Optional[torch.Tensor] = None,        # [N] 节点类型标记
                 **kwargs):
        """
        初始化GKSGraph数据结构
        
        Args:
            nodes: 节点坐标，用于构建物理算子L
            edges: 边连接关系，定义图拓扑
            faces: 面连接关系，定义网格拓扑
            node_features: 时间演化的节点物理量
            source_terms: 时间演化的源项f(x,t)
            initial_condition: 初始条件u0(x)
            time_points: 时间序列点
            edge_attr: 边属性（距离、权重等几何信息）
            boundary_info: 边界条件信息字典，格式：
                {
                    'dirichlet': {  # Dirichlet边界条件（固定值边界）
                        'indices': Tensor,  # [num_dirichlet] 边界节点索引
                        'values': Tensor    # [num_dirichlet] 对应的固定边界值
                    },
                    'neumann': {    # Neumann边界条件（零梯度/绝热边界）
                        'source_indices': Tensor,  # [num_neumann] 内部参考节点索引（提供值）
                        'target_indices': Tensor   # [num_neumann] 边界节点索引（接收值）
                    }
                }
            node_type: 节点类型标记，0=内部节点，1=dirichlet边界，2=neumann边界
            **kwargs: 动态属性支持，可添加任意额外属性
        """
        self.nodes = nodes
        self.edges = edges  
        self.faces = faces
        self.node_features = node_features
        self.source_terms = source_terms
        self.initial_condition = initial_condition
        self.time_points = time_points
        self.edge_attr = edge_attr
        
        # 边界条件支持，节点类型初始化为0
        self.boundary_info = boundary_info or {}
        self.node_type = node_type if node_type is not None else torch.zeros(nodes.shape[0], dtype=torch.long, device=nodes.device)
        
        # 根据边界条件设置节点类型
        # 注意，即使传出的参数已经有节点类型，也会根据边界条件更新节点类型
        # 因此，需要注意在初始化的时候，传入的节点类型跟边界条件一致，否则会覆盖节点类型，导致节点类型标记完全出错
        # 这里code暂时没有写入相应的检查模块，因此处理的时候需要小心，后续会添加检查模块
        if boundary_info:
            self._setup_boundary_conditions(boundary_info)
        
        # 计算基础几何属性，用于构建物理算子
        self._compute_geometric_properties()
        
        # 动态属性添加
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def _setup_boundary_conditions(self, boundary_info: Dict[str, Any]):
        """
        设置边界条件
        
        Args:
            boundary_info: 边界条件信息字典
        """
        # 更新节点类型标记
        for bc_type, bc_data in boundary_info.items():
            if bc_type == 'dirichlet' and 'indices' in bc_data:
                indices = bc_data['indices']
                self.node_type[indices] = 1  # dirichlet边界节点标记为1
            elif bc_type == 'neumann' and 'target_indices' in bc_data:
                target_indices = bc_data['target_indices']
                self.node_type[target_indices] = 2  # neumann边界节点标记为2
    
    def get_boundary_data(self) -> Dict[str, Any]:
        """
        获取边界条件数据
        
        Returns:
            Dict[str, Any]: 边界条件信息字典
        """
        return self.boundary_info
    
    def get_boundary_mask(self, bc_type: str) -> Optional[torch.Tensor]:
        """
        获取指定类型的边界节点掩码
        
        Args:
            bc_type: 边界条件类型 ('dirichlet', 'neumann', 'interior')
            
        Returns:
            torch.Tensor: [N] 布尔掩码，True表示该类型的节点
        """
        if bc_type == 'interior':
            return self.node_type == 0
        elif bc_type == 'dirichlet':
            return self.node_type == 1
        elif bc_type == 'neumann':
            return self.node_type == 2
        elif bc_type == 'boundary':  # 所有边界节点
            return self.node_type > 0
        else:
            return None
    
    def _compute_geometric_properties(self):
        """计算几何属性，为物理算子构建做准备"""

        # 计算并存储节点间距离
        edge_i, edge_j = self.edges[:, 0], self.edges[:, 1]
        edge_vectors = self.nodes[edge_j] - self.nodes[edge_i]
        self.edge_distances = torch.norm(edge_vectors, dim=1)  # [E] 节点间距离，与edges边连接关系对齐
        
        # 计算节点的Voronoi区域体积，用于质量归一化
        self.node_volumes = self._estimate_node_volumes()
    
    # 基于三角剖分计算节点的局域体积/面积（这里使用的是Voronoi区域面积）
    # 每个节点面积为所属相邻三角形面积之和的1/3，而后将所有相邻三角形面积累加
    def _estimate_node_volumes(self):
        """
        基于三角剖分精确计算节点的局域体积/面积。
        用于算子L的构建
        """
        num_nodes = self.nodes.shape[0]
        device = self.nodes.device

        # 提取每个三角形的三个顶点坐标
        # face_vertices shape: [F, 3, spatial_dim]
        face_vertices = self.nodes[self.faces]

        # 计算每个三角形的两条边向量
        # v0, v1, v2 are the vertices of the triangles
        v0, v1, v2 = face_vertices[:, 0, :], face_vertices[:, 1, :], face_vertices[:, 2, :]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # 根据空间维度计算面积
        spatial_dim = self.nodes.shape[1]
        if spatial_dim == 2:
            # 2D情况：使用2D叉乘的模
            # triangle_areas shape: [F]
            triangle_areas = 0.5 * torch.abs(edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0])
        elif spatial_dim == 3:
            # 3D情况：使用3D叉乘的范数
            # triangle_areas shape: [F]
            triangle_areas = 0.5 * torch.norm(torch.cross(edge1, edge2, dim=1), dim=1)
        else:
            raise ValueError(f"Volume calculation for spatial_dim={spatial_dim} not implemented.")

        # 将每个三角形的面积均分给其三个顶点
        area_per_vertex = triangle_areas / 3.0
        
        # 准备scatter_add的源数据
        # src shape: [F*3]
        src = area_per_vertex.repeat_interleave(3)
        
        # 准备scatter_add的目标索引
        # index shape: [F*3]
        index = self.faces.flatten()

        # 初始化节点体积张量
        node_volumes = torch.zeros(num_nodes, dtype=torch.float32, device=device)
        
        # 使用scatter_add_将面积累加到每个节点上
        node_volumes.scatter_add_(0, index, src)
        
        return node_volumes
    
    def get_timestep_data(self, t_idx: int):
        """获取特定时间步的数据（第t_idx个时间步）"""
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'faces': self.faces,
            'node_features': self.node_features[t_idx],
            'source_terms': self.source_terms[t_idx], 
            'time': self.time_points[t_idx],
            'edge_attr': self.edge_attr,
            'node_volumes': self.node_volumes,
            'boundary_info': self.boundary_info,
            'node_type': self.node_type
        }
    
    def get_history_data(self, t_idx: int):
        """获取到时间步t_idx为止的历史数据（包括第t_idx个时间步）"""
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'faces': self.faces,
            'node_features_history': self.node_features[:t_idx+1],
            'source_terms_history': self.source_terms[:t_idx+1],
            'time_history': self.time_points[:t_idx+1],
            'initial_condition': self.initial_condition,
            'edge_attr': self.edge_attr,
            'node_volumes': self.node_volumes,
            'boundary_info': self.boundary_info,
            'node_type': self.node_type
        }

# 数据变换基类，定义统一的变换接口
# BaseTransform类和其子类还没有仔细修正，先注释掉，后续再修改并计入
class BaseTransform(ABC):
    """数据变换基类，定义统一的变换接口"""
    
    @abstractmethod
    def __call__(self, data: DGGraph) -> DGGraph:
        """应用变换到GKSGraph数据"""
        pass

# 归一化变换，对物理量进行标准化处理
# 这里归一化不太准确，后续需要进行无量纲化的归一化，对单个GKSGraph（一个PDE轨迹）进行归一化即可，而不是对整个数据集进行归一化
class Normalize(BaseTransform):
    """归一化变换，对物理量进行标准化处理"""
    
    def __init__(self, 
                 normalize_features: bool = True,
                 normalize_source: bool = True, 
                 normalize_coords: bool = False):
        """
        初始化归一化变换
        
        Args:
            normalize_features: 是否归一化节点特征
            normalize_source: 是否归一化源项
            normalize_coords: 是否归一化坐标
        """
        self.normalize_features = normalize_features
        self.normalize_source = normalize_source  
        self.normalize_coords = normalize_coords
        
        # 统计量将在第一次调用时计算
        self.stats = {}
    
    def __call__(self, data: DGGraph) -> DGGraph:
        """应用归一化变换"""
        data_copy = self._copy_data(data)
        
        if self.normalize_features:
            data_copy.node_features = self._normalize_tensor(
                data_copy.node_features, 'node_features')
        
        if self.normalize_source:
            data_copy.source_terms = self._normalize_tensor(
                data_copy.source_terms, 'source_terms')
                
        if self.normalize_coords:
            data_copy.nodes = self._normalize_tensor(
                data_copy.nodes, 'nodes')
        
        return data_copy
    
    def _normalize_tensor(self, tensor: torch.Tensor, key: str) -> torch.Tensor:
        """对张量进行标准化"""
        if key not in self.stats:
            self.stats[key] = {
                'mean': tensor.mean(),
                'std': tensor.std() + 1e-8
            }
        
        return (tensor - self.stats[key]['mean']) / self.stats[key]['std']
    
    def _copy_data(self, data: DGGraph) -> DGGraph:
        """深拷贝数据结构"""
        # 深拷贝边界条件信息
        boundary_info_copy = {}
        for bc_type, bc_data in data.boundary_info.items():
            boundary_info_copy[bc_type] = {}
            for key, value in bc_data.items():
                if isinstance(value, torch.Tensor):
                    boundary_info_copy[bc_type][key] = value.clone()
                else:
                    boundary_info_copy[bc_type][key] = value
        
        return DGGraph(
            nodes=data.nodes.clone(),
            edges=data.edges.clone(),
            faces=data.faces.clone(),
            node_features=data.node_features.clone(),
            source_terms=data.source_terms.clone(), 
            initial_condition=data.initial_condition.clone(),
            time_points=data.time_points.clone(),
            edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None,
            boundary_info=boundary_info_copy,
            node_type=data.node_type.clone(),
            # 复制其他动态属性
            **{key: getattr(data, key) for key in data.__dict__ 
               if key not in ['nodes', 'edges', 'faces', 'node_features', 'source_terms', 
                             'initial_condition', 'time_points', 'edge_attr', 
                             'boundary_info', 'node_type', 'edge_distances', 'node_volumes']}
        )

class AddNoise(BaseTransform):
    """添加噪声变换，增强模型鲁棒性"""
    
    def __init__(self, 
                 noise_level: float = 0.01,
                 add_to_features: bool = True,
                 add_to_source: bool = True,
                 add_to_initial: bool = True):
        """
        初始化噪声变换
        
        Args:
            noise_level: 噪声强度（相对于数据标准差）
            add_to_features: 是否对节点特征加噪声
            add_to_source: 是否对源项加噪声  
            add_to_initial: 是否对初始条件加噪声
        """
        self.noise_level = noise_level
        self.add_to_features = add_to_features
        self.add_to_source = add_to_source
        self.add_to_initial = add_to_initial
    
    def __call__(self, data: DGGraph) -> DGGraph:
        """应用噪声变换"""
        data_copy = self._copy_data(data)
        
        if self.add_to_features:
            # 环境对齐：使用torch.randn_like直接生成同设备噪声
            noise = torch.randn_like(data_copy.node_features) * self.noise_level
            data_copy.node_features += noise * data_copy.node_features.std()
        
        if self.add_to_source:
            noise = torch.randn_like(data_copy.source_terms) * self.noise_level  
            data_copy.source_terms += noise * data_copy.source_terms.std()
            
        if self.add_to_initial:
            noise = torch.randn_like(data_copy.initial_condition) * self.noise_level
            data_copy.initial_condition += noise * data_copy.initial_condition.std()
        
        return data_copy
    
    def _copy_data(self, data: DGGraph) -> DGGraph:
        """深拷贝数据结构"""
        # 深拷贝边界条件信息
        boundary_info_copy = {}
        for bc_type, bc_data in data.boundary_info.items():
            boundary_info_copy[bc_type] = {}
            for key, value in bc_data.items():
                if isinstance(value, torch.Tensor):
                    boundary_info_copy[bc_type][key] = value.clone()
                else:
                    boundary_info_copy[bc_type][key] = value
        
        return DGGraph(
            nodes=data.nodes.clone(),
            edges=data.edges.clone(),
            faces=data.faces.clone(),
            node_features=data.node_features.clone(),
            source_terms=data.source_terms.clone(), 
            initial_condition=data.initial_condition.clone(),
            time_points=data.time_points.clone(),
            edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None,
            boundary_info=boundary_info_copy,
            node_type=data.node_type.clone(),
            # 复制其他动态属性
            **{key: getattr(data, key) for key in data.__dict__ 
               if key not in ['nodes', 'edges', 'faces', 'node_features', 'source_terms', 
                             'initial_condition', 'time_points', 'edge_attr', 
                             'boundary_info', 'node_type', 'edge_distances', 'node_volumes']}
        )

class Compose(BaseTransform):
    """组合多个变换"""
    
    def __init__(self, transforms: List[BaseTransform]):
        """
        初始化组合变换
        
        Args:
            transforms: 变换列表，按顺序应用
        """
        self.transforms = transforms
        
    def __call__(self, data: DGGraph) -> DGGraph:
        """按顺序应用所有变换"""
        for transform in self.transforms:
            data = transform(data)
        return data

class DGPdeDataset(Dataset):
    """
    GKSNet的PyTorch数据集类
    
    从HDF5文件中加载PDE仿真轨迹数据，并将其封装为GKSGraph对象
    核心功能是根据配置将长轨迹切分为多个短轨迹储存在内存中，用于高效训练
    """
    
    def __init__(self, 
                 data_path: str,
                 train_time_steps: int,
                 max_samples: Optional[int] = None,
                 rank: int = 0,
                 trajectory_keys: Optional[List[str]] = None):
        """
        初始化数据集
        
        Args:
            data_path: HDF5数据文件路径
            train_time_steps: 用于轨迹切分的子轨迹时间步长度 M
            max_samples: 可选的最大样本数
            rank: DDP进程ID，用于控制打印
            trajectory_keys: 可选的轨迹键列表，若提供则只加载这些轨迹
        """
        self.data_path = data_path
        self.train_time_steps = train_time_steps
        self.samples = []  # 用于存储所有切分后的短轨迹样本 (GKSGraph)
        self.rank = rank
        self.trajectory_keys = trajectory_keys
        
        self._chunk_and_load_data()
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _chunk_and_load_data(self):
        """
        核心函数：加载所有轨迹，将其切分为短轨迹块，并存储在内存中
        """
        # 从配置中获取训练时间步 M
        if not self.train_time_steps or self.train_time_steps <= 0:
            raise ValueError("参数 'train_time_steps' 必须是一个正整数")

        if self.rank == 0:
            print(f"数据加载：将使用 'train_time_steps={self.train_time_steps}' 对轨迹进行切分...")

        # 打开数据文件，并获取数据集基本信息
        with h5py.File(self.data_path, 'r') as f:

            # 获取所有轨迹的key
            traj_keys = sorted(list(f.keys()))
            if self.trajectory_keys is not None:
                missing = [k for k in self.trajectory_keys if k not in f]
                if missing:
                    raise KeyError(f"HDF5中未找到轨迹: {missing}")
                traj_keys = [k for k in traj_keys if k in self.trajectory_keys]
            
            # 遍历所有轨迹
            for traj_key in traj_keys:
                # 获取当前轨迹的组
                traj_group = f[traj_key]
                
                # 从HDF5加载完整轨迹的通用数据
                nodes = torch.from_numpy(traj_group['nodes'][:]).float() # [T, N, spatial_dim] 节点坐标
                edges = torch.from_numpy(traj_group['edges'][:]).long() # [E, 2] 边连接关系
                faces = torch.from_numpy(traj_group['faces'][:]).long() # [F, 3] 面连接关系
                
                # 从HDF5加载完整轨迹的物理量数据
                full_node_features = torch.from_numpy(traj_group['node_features'][:]).float() # [T, N, feature_dim] 节点特征
                full_source_terms = torch.from_numpy(traj_group['source_terms'][:]).float() # [T, N, source_dim] 源项
                full_time_points = torch.from_numpy(traj_group['time_points'][:]).float() # [T] 时间点

                # 加载边界条件信息
                boundary_info = {}
                if 'boundary_info' in traj_group:
                    bc_group = traj_group['boundary_info']
                    if 'dirichlet' in bc_group:
                        dirichlet_group = bc_group['dirichlet']
                        boundary_info['dirichlet'] = {
                            'indices': torch.from_numpy(dirichlet_group['indices'][:]).long(),
                            'values': torch.from_numpy(dirichlet_group['values'][:]).float()
                        }
                    if 'neumann' in bc_group:
                        neumann_group = bc_group['neumann']
                        boundary_info['neumann'] = {
                            'source_indices': torch.from_numpy(neumann_group['source_indices'][:]).long(),
                            'target_indices': torch.from_numpy(neumann_group['target_indices'][:]).long(),
                        }
                
                # 对长轨迹进行切分
                T = full_node_features.shape[0]  # 完整轨迹的总时间步
                M = self.train_time_steps        # 子轨迹的长度
                num_chunks = T // M              # 计算可以切分出多少个完整的块

                # 遍历所有块
                for i in range(num_chunks):

                    # 计算当前块的开始和结束索引
                    start_idx = i * M
                    end_idx = start_idx + M
                    
                    # 提取当前块的数据
                    chunk_node_features = full_node_features[start_idx:end_idx]
                    chunk_source_terms = full_source_terms[start_idx:end_idx]
                    chunk_time_points = full_time_points[start_idx:end_idx]
                    
                    # 子轨迹的初始条件是该块的第一个时间步的状态
                    chunk_initial_condition = full_node_features[start_idx]
                    
                # 为当前块创建一个 GKSGraph 实例
                    dg_graph_chunk = DGGraph(
                        nodes=nodes,
                        edges=edges,
                        faces=faces,
                        node_features=chunk_node_features,
                        source_terms=chunk_source_terms,
                        initial_condition=chunk_initial_condition,
                        time_points=chunk_time_points,
                        boundary_info=boundary_info,
                    trajectory_id=traj_key,
                    )
                    
                    self.samples.append(dg_graph_chunk)
        
        if self.rank == 0:
            print(f"数据加载完成：共生成 {len(self.samples)} 个短轨迹样本。")

    def __len__(self) -> int:
        """返回数据集中短轨迹样本的总数"""
        return len(self.samples)

    def __getitem__(self, index: int) -> DGGraph:
        """获取一个经过切分的短轨迹样本"""
        return self.samples[index]


def dg_collate_fn(batch_list: List[DGGraph]) -> Dict[str, Any]:
    """
    自定义批处理函数，将多个GKSGraph合并为一个批次字典
    对于GKSNets，我们需要保持每个图的独立性，同时支持batch处理，因此需要自定义批处理函数
    
    Args:
        batch_list: GKSGraph对象列表
        
    Returns:
        Dict: 批处理后的数据字典
    """
    # 获取批次大小
    batch_size = len(batch_list)
    
    # 这里假设批次中所有图具有相同的几何结构和边界条件，取第一个图作为样本
    # 后续看情况可能需要修改，比如取多个几何结构不一致的图作为样本
    sample = batch_list[0]
    
    # 创建批次字典
    batch_data = {
        'batch_size': batch_size, # 批次大小B
        'num_nodes': sample.nodes.shape[0], # 节点数量N
        'num_timesteps': sample.node_features.shape[0], # 时间步数T
        
        # 几何信息（假设所有图共享相同几何，即节点坐标、边连接关系、边属性、节点体积）
        'nodes': sample.nodes,  # [N, spatial_dim] 节点坐标
        'edges': sample.edges,  # [E, 2] 边连接关系
        'faces': sample.faces,  # [F, 3] 面连接关系
        'edge_attr': sample.edge_attr,  # [E, edge_dim] or None 边属性（取决于输入）
        'node_volumes': sample.node_volumes,  # [N] 节点体积
        
        # 批次数据
        'node_features': torch.stack([g.node_features for g in batch_list]),  # [B, T, N, feature_dim] 节点特征
        'source_terms': torch.stack([g.source_terms for g in batch_list]),   # [B, T, N, source_dim] 源项
        'initial_conditions': torch.stack([g.initial_condition for g in batch_list]), # [B, N, state_dim] 初始条件
        'time_points': sample.time_points,  # [T] 时间点
        
        # 边界条件信息（假设所有图共享相同边界条件）
        'boundary_info': sample.boundary_info,  # 边界条件数据，格式为边界条件信息字典，包含dirichlet和neumann两种边界条件，与之前一致
        'node_type': sample.node_type,  # [N] 节点类型标记
        
        # 批次中样本的轨迹ID
        'trajectory_ids': [getattr(g, 'trajectory_id', i) for i, g in enumerate(batch_list)] # [B] 轨迹ID
    }
    
    return batch_data

# 创建数据加载器，用于加载数据集
def create_dg_loader(dataset: DGPdeDataset,
                     batch_size: int = 32,
                     shuffle: bool = True,
                     num_workers: int = 8,
                     pin_memory: bool = True,
                     sampler: Optional[torch.utils.data.Sampler] = None,
                     **kwargs) -> DataLoader:
    """
    创建GKSNets数据加载器
    
    Args:
        dataset: GKSPdeDataset实例
        batch_size: 批大小
        shuffle: 是否随机打乱 (在使用sampler时应设为False)
        num_workers: 并行加载进程数
        pin_memory: 是否固定内存
        sampler: 自定义采样器 (例如 DistributedSampler)
        **kwargs: 其他DataLoader参数
        
    Returns:
        DataLoader: 配置好的数据加载器
    """
    
    # 如果提供了sampler，shuffle必须为False
    if sampler is not None:
        shuffle = False

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=dg_collate_fn,
        sampler=sampler,
        **kwargs
    )
