"""
GKSNets物理计算核心模块
包含物理算子构建、Green函数核计算、边界条件处理等功能
"""

import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from typing import Tuple, Optional, Dict, Any
import warnings


def build_operator(nodes: torch.Tensor, 
                  edges: torch.Tensor,
                  faces: torch.Tensor,
                  node_volumes: torch.Tensor,
                  operator_type: str = 'laplace',
                  edge_attr: Optional[torch.Tensor] = None,
                  **kwargs) -> torch.Tensor:
    """
    根据几何信息构建离散的物理算子矩阵L_base
    
    这是GKSNets方法的物理基础，将连续的微分算子离散化为图矩阵
    支持不同类型的算子：拉普拉斯、对流、反应等
    
    Args:
        nodes: [N, spatial_dim] 节点坐标
        edges: [E, 2] 边连接关系
        faces: [F, 3] 面（三角形）连接关系
        node_volumes: [N] 节点的局域体积/面积 (用于质量归一化)
        operator_type: 算子类型 ('laplace', 'advection', 'reaction')
        edge_attr: [E, edge_dim] 可选的边属性（只要提供了该参数，默认最开始的属性就是预计算的权重）
        **kwargs: 算子特定的参数
        
    Returns:
        torch.Tensor: [N, N] 离散算子矩阵L_base
        
    Note:
        这个函数实现您方法中的基础物理算子L_base，算子矩阵乘以列向量来作用于列向量
        而后会通过位置A的GNN进行修正，修正后的算子矩阵为L_base + ΔL
    """
    N = nodes.shape[0]
    device = nodes.device
    
    # 初始化算子矩阵
    L = torch.zeros(N, N, dtype=torch.float32, device=device)
    
    if operator_type == 'laplace':
        L = _build_laplace_operator(nodes, edges, faces, node_volumes, edge_attr, **kwargs)
    elif operator_type == 'gradient':
        L = _build_gradient_operator(nodes, faces, **kwargs)
    elif operator_type == 'gradient_gauss':
        L = _build_gradient_operator_gauss(nodes, edges, faces, node_volumes, **kwargs)
    elif operator_type == 'fhn':
        L = _build_fhn_operator(nodes, edges, faces, node_volumes, edge_attr, **kwargs)
    # elif operator_type == 'advection':
    #     L = _build_advection_operator(nodes, edges, node_volumes, edge_attr, **kwargs)  
    # elif operator_type == 'reaction':
    #     L = _build_reaction_operator(nodes, edges, node_volumes, edge_attr, **kwargs)
    else:
        raise ValueError(f"Unsupported operator type: {operator_type}")
    
    return L

# 拉普拉斯算子构建函数
# 这里构建的拉普拉斯算子是基于离散的拉普拉斯-贝尔特拉米算子，适用于不规则网格
# 拉普拉斯算子矩阵乘以列向量来作用于列向量，即L*u，其中u是状态向量，L是拉普拉斯算子矩阵
def _build_laplace_operator(nodes: torch.Tensor,
                           edges: torch.Tensor,
                           faces: torch.Tensor,
                           node_volumes: torch.Tensor,
                           edge_attr: Optional[torch.Tensor] = None,
                           **kwargs) -> torch.Tensor:
    """
    构建拉普拉斯-贝尔特拉米算子 ∇²u
    
    使用离散拉普拉斯-贝尔特拉米算子，适用于不规则网格。
    对于内部边 (i,j)，其权重 w_ij = 0.5 * (cot(α_ij) + cot(β_ij))，其中α和β为该边对角。
    
    Args:
        nodes: [N, spatial_dim] 节点坐标
        edges: [E, 2] 边连接关系（无向边）
        faces: [F, 3] 面（三角形）连接关系
        node_volumes: [N] 节点的局域体积/面积，用于质量归一化
        edge_attr: [E, edge_dim] 可选的边属性。如果提供，将跳过cotangent权重计算，直接使用此预计算权重。
        
    Returns:
        torch.Tensor: [N, N] 拉普拉斯算子矩阵L
    """
    N = nodes.shape[0]
    device = nodes.device
    
    if edge_attr is not None:
        # 如果提供了预计算的权重，则直接使用
        weights = edge_attr[:, 0] if edge_attr.dim() > 1 else edge_attr
    else:
        # 基于余切公式（Cotangent Formula）计算边权重，cot_values是每个三角形三个角的cot值
        face_vertices = nodes[faces]
        v0, v1, v2 = face_vertices[:, 0, :], face_vertices[:, 1, :], face_vertices[:, 2, :]

        # 计算每个三角形边的向量
        vec_01, vec_12, vec_20 = v1 - v0, v2 - v1, v0 - v2
        
        # 计算每个三角形三个角的余切值
        # cot(angle) = (a·b) / ||a x b||
        dot_p2 = torch.sum(-vec_20 * vec_12, dim=1) # 角 at v2
        dot_p0 = torch.sum(-vec_01 * vec_20, dim=1) # 角 at v0
        dot_p1 = torch.sum(-vec_12 * vec_01, dim=1) # 角 at v1

        # 计算面积的两倍 (2D/3D通用)
        if nodes.shape[1] == 2:
            two_areas = torch.abs(vec_01[:, 0] * vec_12[:, 1] - vec_01[:, 1] * vec_12[:, 0])
        else: # 3D
            two_areas = torch.norm(torch.cross(vec_01, vec_12, dim=1), dim=1)
        
        # 确保面积不为零
        two_areas = two_areas.clamp(min=1e-8)
        
        # cot_p0, cot_p1, cot_p2 分别是顶点 v0, v1, v2 对角的cot值
        cot_values = torch.stack([dot_p0 / two_areas, dot_p1 / two_areas, dot_p2 / two_areas], dim=1)

        # i, j, k 是每个face的三个顶点索引
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
        
        # cot_pk 是边(i,j)的对角cot值, cot_pi是边(j,k)的, cot_pj是边(k,i)的
        cot_pk, cot_pi, cot_pj = cot_values[:, 2], cot_values[:, 0], cot_values[:, 1]
        
        # 使用稀疏矩阵累加所有边的cot值
        indices = torch.cat([torch.stack([i, j]), torch.stack([j, i]),
                             torch.stack([j, k]), torch.stack([k, j]),
                             torch.stack([k, i]), torch.stack([i, k])], dim=1)
        
        # 将cot_pk, cot_pi, cot_pj三个角对应的cot值拼接起来
        values = torch.cat([cot_pk, cot_pk, cot_pi, cot_pi, cot_pj, cot_pj])
        
        # 聚合每个边的cot值之和
        L_cot_sparse = torch.sparse_coo_tensor(indices, values, (N, N), device=device).coalesce()
        
        # 转换为稠密矩阵以进行索引 (修复SparseCUDA索引错误)
        L_cot_dense = L_cot_sparse.to_dense()

        # 从稠密矩阵中提取原始 `edges` 对应的权重
        # L_cot_dense[i, j] 包含了两个对角的cot之和
        weights = L_cot_dense[edges[:, 0], edges[:, 1]] * 0.5

    # 初始化拉普拉斯矩阵
    L = torch.zeros(N, N, device=device)
    
    # 构建算子矩阵
    for edge_idx, (i, j) in enumerate(edges):
        i, j = int(i), int(j)
        
        # 获取边(i,j)的权重
        w_ij = weights[edge_idx]
        
        # 质量归一化
        w_ij_normalized_i = w_ij / node_volumes[i]
        w_ij_normalized_j = w_ij / node_volumes[j]
        
        # 构建拉普拉斯矩阵
        L[i, j] = -w_ij_normalized_i
        L[j, i] = -w_ij_normalized_j
    
    # 设置对角元，确保行和为零（Laplace算子的守恒性质）
    for i in range(N):
        L[i, i] = -torch.sum(L[i, :])
    
    # 定义物理常数
    conductivity = 50.0  # 热导率 k (W/m*K)
    rho = 7850.0         # 密度 (kg/m^3)
    specific_heat = 450.0  # 比热 (J/kg*K)

    # 计算归一化后的扩散系数 alpha = k / (rho * c)
    diffusion_coeff = conductivity / (rho * specific_heat)
    
    return - diffusion_coeff * L

def _build_gradient_operator(nodes: torch.Tensor,
                             faces: torch.Tensor,
                             **kwargs) -> torch.Tensor:
    """
    构建一个基于面积加权平均的离散梯度算子矩阵。
    
    这个实现遵循了您的要求，返回 Dx + Dy 两个偏导算子矩阵的和。
    注意：这会产生一个标量场算子 ∂f/∂x + ∂f/∂y，而非梯度矢量场 (∂f/∂x, ∂f/∂y)。

    Args:
        nodes: [N, 2] 节点坐标。目前只支持2D网格。
        faces: [F, 3] 三角形面连接关系。

    Returns:
        torch.Tensor: [N, N] 梯度算子矩阵 (Dx + Dy)。
    """
    N = nodes.shape[0]
    F = faces.shape[0]
    device = nodes.device

    if nodes.shape[1] != 2:
        warnings.warn("Gradient operator is only implemented for 2D meshes. Returning a zero matrix.")
        return torch.zeros(N, N, device=device)

    # 初始化算子矩阵和用于归一化的节点周围总面积
    Dx = torch.zeros(N, N, device=device)
    Dy = torch.zeros(N, N, device=device)
    node_total_areas = torch.zeros(N, device=device)

    # 提取所有面的顶点坐标
    # face_vertices shape: [F, 3, 2]
    face_vertices = nodes[faces]
    v0, v1, v2 = face_vertices[:, 0, :], face_vertices[:, 1, :], face_vertices[:, 2, :]

    # 计算每个三角形的边向量和面积的两倍
    # 面积 = 0.5 * |x1*y2 - x2*y1|
    two_areas = ( (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - 
                  (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0]) ).abs()
    
    # 为防止除以零，将过小的面积设置为一个很小的值
    two_areas = two_areas.clamp(min=1e-12)
    areas = two_areas / 2.0

    # 预先计算每个三角形内，梯度对三个顶点函数值的贡献系数
    # (∂f/∂x)_T = c_x0*f0 + c_x1*f1 + c_x2*f2
    c_x0 = (v1[:, 1] - v2[:, 1]) / two_areas
    c_x1 = (v2[:, 1] - v0[:, 1]) / two_areas
    c_x2 = (v0[:, 1] - v1[:, 1]) / two_areas
    # (∂f/∂y)_T = c_y0*f0 + c_y1*f1 + c_y2*f2
    c_y0 = (v2[:, 0] - v1[:, 0]) / two_areas
    c_y1 = (v0[:, 0] - v2[:, 0]) / two_areas
    c_y2 = (v1[:, 0] - v0[:, 0]) / two_areas

    # 累加每个节点周围的总面积，用于后续的归一化
    # faces.flatten() 将所有顶点索引展平
    # areas.repeat_interleave(3) 将每个面积重复3次，以匹配展平的顶点
    node_total_areas.scatter_add_(0, faces.flatten(), areas.repeat_interleave(3))

    # 遍历每个面，将其对三个顶点的梯度贡献累加到Dx和Dy矩阵中
    for f_idx in range(F):
        # 获取当前面的三个顶点索引
        i, j, k = faces[f_idx]
        
        # 获取该面梯度对于 f_i, f_j, f_k 的系数
        c_face_x = torch.tensor([c_x0[f_idx], c_x1[f_idx], c_x2[f_idx]], device=device)
        c_face_y = torch.tensor([c_y0[f_idx], c_y1[f_idx], c_y2[f_idx]], device=device)
        
        # 将这个面（face）的梯度贡献分配给它的三个顶点（vertex）
        # 节点 i 的梯度，是周围所有面梯度的面积加权平均。
        # 这里先累加分子部分：Area * (Face Gradient Coeffs)
        # 权重是当前面的面积
        area_weight = areas[f_idx]
        
        # 对顶点i (row i)的贡献
        Dx[i, [i,j,k]] += area_weight * c_face_x
        Dy[i, [i,j,k]] += area_weight * c_face_y

        # 对顶点j (row j)的贡献
        Dx[j, [i,j,k]] += area_weight * c_face_x
        Dy[j, [i,j,k]] += area_weight * c_face_y

        # 对顶点k (row k)的贡献
        Dx[k, [i,j,k]] += area_weight * c_face_x
        Dy[k, [i,j,k]] += area_weight * c_face_y
        
    # 归一化：将Dx和Dy的每一行除以对应节点的总面积
    # unsqueeze(1) 用于广播
    Dx = Dx / node_total_areas.unsqueeze(1).clamp(min=1e-12)
    Dy = Dy / node_total_areas.unsqueeze(1).clamp(min=1e-12)

    return -0.05 * Dx


def _build_gradient_operator_gauss(nodes: torch.Tensor,
                                   edges: torch.Tensor,
                                   faces: torch.Tensor,
                                   node_volumes: torch.Tensor,
                                   **kwargs) -> torch.Tensor:
    """
    使用格林-高斯方法构建更高精度的离散梯度算子矩阵。

    该方法基于控制体积边界的通量积分，通常比基于单元梯度的平均法更鲁棒、精度更高。
    它不涉及任何矩阵求逆，是完全的构造性方法。

    Args:
        nodes: [N, 2] 节点坐标。目前只支持2D网格。
        edges: [E, 2] 无向、无重合的边连接关系。
        faces: [F, 3] 无重合的面连接关系。
        node_volumes: [N] 每个节点的对偶控制体积面积，用于归一化。

    Returns:
        torch.Tensor: [N, N] 梯度算子矩阵 (-0.1 * Dx - 0.12 * Dy)。
    """
    N = nodes.shape[0]
    device = nodes.device

    if nodes.shape[1] != 2:
        warnings.warn("Green-Gauss gradient operator is only implemented for 2D meshes. Returning a zero matrix.")
        return torch.zeros(N, N, device=device)

    # --- 1. 预计算 ---

    # 计算所有三角形的形心
    face_centroids = nodes[faces].mean(dim=1)

    # 构建边到共享面的映射，这是实现格林-高斯法的关键数据结构
    edge_to_faces_map = {}
    for face_idx, face in enumerate(faces):
        v = face.tolist()
        # 为保证边的唯一性，使用排序后的顶点元组作为键
        edge1 = tuple(sorted((v[0], v[1])))
        edge2 = tuple(sorted((v[1], v[2])))
        edge3 = tuple(sorted((v[2], v[0])))
        
        edge_to_faces_map.setdefault(edge1, []).append(face_idx)
        edge_to_faces_map.setdefault(edge2, []).append(face_idx)
        edge_to_faces_map.setdefault(edge3, []).append(face_idx)

    # --- 2. 构造系数矩阵 ---

    # 初始化算子矩阵
    Dx = torch.zeros(N, N, device=device)
    Dy = torch.zeros(N, N, device=device)

    # 遍历每一条边，计算其对相邻节点梯度的贡献
    for edge_tuple, face_indices in edge_to_faces_map.items():
        # 格林-高斯法主要应用于内部边，边界边可采用特殊处理，这里为简化，仅处理内部边
        if len(face_indices) != 2:
            continue

        # 获取边的两个顶点 i, j 和共享此边的两个三角形的形心 C1, C2
        i, j = edge_tuple
        face1_idx, face2_idx = face_indices
        C1 = face_centroids[face1_idx]
        C2 = face_centroids[face2_idx]

        # 计算分隔节点i和j的控制体积边界面的矢量面积 S_ij
        # 矢量方向从C1指向C2，再旋转90度得到法向，即 (dy, -dx)
        # 这里 S_ij 是从 V_i 指向 V_j 的法向矢量面积
        S_ij = torch.tensor([C2[1] - C1[1], C1[0] - C2[0]], device=device)

        # 根据公式 grad(f)_i = (1/Area_i) * sum(f_face * S_face)
        # f_face 约等于 (f_i + f_j) / 2
        # 提取 f_i 和 f_j 的系数
        # 对 grad(f)_i, f_i 的系数是 S_ij/2, f_j 的系数也是 S_ij/2
        # 对 grad(f)_j, 系数相反，因为 S_ji = -S_ij
        
        # 累加系数到Dx和Dy矩阵中，此时尚未归一化
        # 对节点 i 的梯度贡献
        Dx[i, i] += S_ij[0] * 0.5
        Dy[i, i] += S_ij[1] * 0.5
        Dx[i, j] += S_ij[0] * 0.5
        Dy[i, j] += S_ij[1] * 0.5

        # 对节点 j 的梯度贡献 (法向量相反)
        Dx[j, j] -= S_ij[0] * 0.5
        Dy[j, j] -= S_ij[1] * 0.5
        Dx[j, i] -= S_ij[0] * 0.5
        Dy[j, i] -= S_ij[1] * 0.5

    # --- 3. 归一化 ---
    # 将每一行除以对应节点的控制体积面积
    # 添加一个很小的值避免除以零
    safe_node_volumes = node_volumes.unsqueeze(1).clamp(min=1e-12)
    Dx = Dx / safe_node_volumes
    Dy = Dy / safe_node_volumes

    # --- 4. 组合成最终的对流算子 ---
    return -0.05 * Dx


def _build_fhn_operator(nodes: torch.Tensor,
                        edges: torch.Tensor,
                        faces: torch.Tensor,
                        node_volumes: torch.Tensor,
                        edge_attr: Optional[torch.Tensor] = None,
                        **kwargs) -> torch.Tensor:
    """
    构建一个用于菲茨休-南云(FHN)方程的组合算子。

    该算子是拉普拉斯算子(扩散项)和梯度算子(平流项)的和, 代表: ∇²u - c·∇u
    这里扩散系数为1, 平流速度 c 在梯度算子中硬编码。

    Args:
        nodes: [N, 2] 节点坐标。
        edges: [E, 2] 无向、无重合的边连接关系。
        faces: [F, 3] 无重合的面连接关系。
        node_volumes: [N] 每个节点的对偶控制体积面积，用于归一化。
        edge_attr: 可选的边属性。
        **kwargs: 传递给子算子构建函数的其他参数。

    Returns:
        torch.Tensor: [N, N] FHN组合算子矩阵。
    """
    # 构建拉普拉斯算子部分 (∇²)
    laplace_op = _build_laplace_operator(nodes, edges, faces, node_volumes, edge_attr, **kwargs)

    # 构建梯度算子部分 (-c·∇)
    # 注意: _build_gradient_operator_gauss 内部已硬编码了平流速度
    gradient_op = _build_gradient_operator_gauss(nodes, edges, faces, node_volumes, **kwargs)

    # 直接返回两者的和, 代表 ∇²u - c·∇u
    return laplace_op + gradient_op


def apply_bcs_to_state(u: torch.Tensor, 
                      bc_data: Dict[str, Any]) -> torch.Tensor:
    """
    对物理状态向量应用边界条件
    
    用于在物理路径u_green和数据路径u_net的输出上强制边界约束
    
    Args:
        u: [N] 或 [B, N] ，分别为单个状态张量和批量状态张量
        bc_data: 边界条件数据字典，包含以下可选键：
                'dirichlet': Dirichlet边界条件（固定值边界）
                    - 'indices': [num_dirichlet] 需要固定值的边界节点索引
                    - 'values': [num_dirichlet] 对应的固定边界值
                'neumann': Neumann边界条件（零梯度/绝热边界）
                    - 'source_indices': [num_neumann] 提供参考值的内部节点索引（被复制的索引）
                    - 'target_indices': [num_neumann] 接收参考值的边界节点索引（复制后粘贴的索引）
        
    Returns:
        torch.Tensor: 应用边界条件后的状态向量
    """
    u_corrected = u.clone()
    
    # 应用Dirichlet边界条件 
    if 'dirichlet' in bc_data:
        dirichlet_bc = bc_data['dirichlet']
        u_corrected = _apply_dirichlet(u_corrected, 
                                     dirichlet_bc['indices'], 
                                     dirichlet_bc['values'])
    
    # 应用Neumann边界条件
    if 'neumann' in bc_data:
        neumann_bc = bc_data['neumann']
        u_corrected = _apply_neumann(u_corrected,
                                   neumann_bc['source_indices'],
                                   neumann_bc['target_indices'])
    
    return u_corrected


def apply_bcs_to_hidden_state(h: torch.Tensor,
                             bc_data: Dict[str, Any]) -> torch.Tensor:
    """
    在GNN消息传递过程中对隐藏状态应用边界条件
    
    确保GNN的中间表示也满足边界约束
    
    Args:
        h: [N, hidden_dim] 或 [B, N, hidden_dim]，分别表示单个图节点隐藏状态张量和批量图节点隐藏状态张量
        bc_data: 边界条件数据字典，和apply_bcs_to_state的bc_data相同
        
    Returns:
        torch.Tensor: 应用边界条件后的隐藏状态
    """
    # 对隐藏状态的每个维度独立应用边界条件（即修改边界特征的每个维度上的值）
    if h.dim() == 2:  # [N, hidden_dim]
        h_corrected = h.clone()
        for dim in range(h.shape[1]):
            h_corrected[:, dim] = apply_bcs_to_state(h[:, dim], bc_data)
    elif h.dim() == 3:  # [B, N, hidden_dim]
        h_corrected = h.clone()
        for batch_idx in range(h.shape[0]):
            for dim in range(h.shape[2]):
                h_corrected[batch_idx, :, dim] = apply_bcs_to_state(
                    h[batch_idx, :, dim], bc_data)
    else:
        raise ValueError(f"Unsupported hidden state dimension: {h.dim()}")
    
    return h_corrected


def _apply_dirichlet(tensor: torch.Tensor, 
                    indices: torch.Tensor, 
                    values: torch.Tensor) -> torch.Tensor:
    """
    实现Dirichlet边界条件：u[boundary] = prescribed_value
    
    Args:
        tensor: [N] 或 [B, N] ，分别为单个状态张量和批量状态张量
        indices: [num_bc] 边界节点索引
        values: [num_bc] 边界值
        
    Returns:
        torch.Tensor: 应用边界条件后的张量，形状与输入tensor相同
    """
    tensor_corrected = tensor.clone()
    
    # 支持批处理
    if tensor.dim() == 1:  # [N]
        tensor_corrected[indices] = values
    elif tensor.dim() == 2:  # [B, N] 
        tensor_corrected[:, indices] = values.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported tensor dimension for Dirichlet BC: {tensor.dim()}")
    
    return tensor_corrected


def _apply_neumann(tensor: torch.Tensor,
                  source_indices: torch.Tensor,
                  target_indices: torch.Tensor) -> torch.Tensor:
    """
    实现Neumann边界条件：∂u/∂n = prescribed_flux
    
    通过设置边界节点值等于相邻内部节点值来近似零梯度条件
    
    Args:
        tensor: [N] 或 [B, N] ，分别为单个状态张量和批量状态张量
        source_indices: [num_bc] 内部节点索引
        target_indices: [num_bc] 边界节点索引
        
    Returns:
        torch.Tensor: 应用边界条件后的张量
    """
    tensor_corrected = tensor.clone()
    
    # 支持批处理
    if tensor.dim() == 1:  # [N]
        tensor_corrected[target_indices] = tensor_corrected[source_indices]
    elif tensor.dim() == 2:  # [B, N]
        tensor_corrected[:, target_indices] = tensor_corrected[:, source_indices]
    else:
        raise ValueError(f"Unsupported tensor dimension for Neumann BC: {tensor.dim()}")
    
    return tensor_corrected

