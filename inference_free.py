"""
GKSNet 推理脚本

功能:
1. 加载一个训练好的 GKSNet 模型检查点。
2. 从数据集中选择一条完整的PDE轨迹。
3. 使用轨迹的初始条件作为起点，以 Free-Running (自回归) 模式进行单步推理，直到最后一个时间步。
4. 可视化对比：在选定的时间步，并排展示真实解 (Ground Truth) 和模型预测 (Prediction)。
"""

import os
# 指定GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib as mpl
from matplotlib.colors import SymLogNorm, LinearSegmentedColormap
import argparse
import pathlib
from typing import Dict, Any, List

# 导入GKSNet模块
from gks_net import GKSNet
# from physics import build_operator, GreenKernelCalculator # No longer needed
from dataset import GKSGraph # 仅用于类型提示和结构参考

def compute_state_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算预测与目标之间的相对误差"""
    with torch.no_grad():
        # 确保张量为浮点型以进行范数计算
        pred = pred.float()
        target = target.float()
        # 避免当目标范数为零时除以零
        target_norm = torch.norm(target)
        if target_norm == 0:
            return 0.0 if torch.norm(pred) == 0 else float('inf')
        error = torch.norm(pred - target) / target_norm
        return error.item()

def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> GKSNet:
    """从检查点加载模型"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    print(f"正在从 '{checkpoint_path}' 加载模型...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = GKSNet(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # 设置为评估模式
    
    print("模型加载成功。")
    return model

def get_trajectory_data(data_path: str, trajectory_key: str) -> Dict[str, np.ndarray]:
    """从HDF5文件加载单条轨迹数据"""
    print(f"正在从 '{data_path}' 加载轨迹 '{trajectory_key}'...")
    with h5py.File(data_path, 'r') as f:
        if trajectory_key not in f:
            raise KeyError(f"在HDF5文件中未找到轨迹: {trajectory_key}")
        
        traj_group = f[trajectory_key]
        
        data = {
            'nodes': traj_group['nodes'][:],
            'edges': traj_group['edges'][:],
            'faces': traj_group['faces'][:],
            'node_features': traj_group['node_features'][:],
            'source_terms': traj_group['source_terms'][:],
            'initial_condition': traj_group['initial_condition'][:],
            'time_points': traj_group['time_points'][:]
        }

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
        data['boundary_info'] = boundary_info
        
    print("轨迹数据加载完成。")
    return data

def visualize_single_snapshot(nodes, ground_truth_t, prediction_t, time_point_t, output_filename, loss, error, faces):
    """
    可视化单个时间步的对比图并保存。
    采用每帧独立的线性标准化来清晰对比。
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), squeeze=False)
    fig.suptitle(f"Inference at t={time_point_t:.3f}s (Per-Frame Linear Norm)", fontsize=16)

    triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=faces)

    # --- 采用基于物理的染色方案 (类似generate_laser_data.py) ---
    T_ambient = 298.15  # 环境温度作为基准

    # 结合真实值和预测值，计算一个鲁棒的颜色上限
    combined_data = np.concatenate([ground_truth_t, prediction_t])
    heated_data = combined_data[combined_data > T_ambient + 1.0] # 仅考虑被加热的区域
    if heated_data.size > 0:
        vmax = np.percentile(heated_data, 99.9)
    else:
        vmax = T_ambient + 1.0 # 如果没有明显加热，则设置一个默认值

    vmin = T_ambient
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # --- 左图: Ground Truth ---
    ax_gt = axes[0, 0]
    mappable = ax_gt.tripcolor(triangulation, ground_truth_t, cmap='hot', norm=norm, shading='gouraud')
    ax_gt.set_title("Ground Truth")
    ax_gt.set_xlabel("x")
    ax_gt.set_ylabel("y")
    ax_gt.set_aspect('equal', adjustable='box')

    # --- 右图: Prediction ---
    ax_pred = axes[0, 1]
    ax_pred.tripcolor(triangulation, prediction_t, cmap='hot', norm=norm, shading='gouraud')
    ax_pred.set_title("Prediction")
    ax_pred.set_xlabel("x")
    ax_pred.set_ylabel("") # Hide y-axis label
    ax_pred.set_yticklabels([])
    ax_pred.set_aspect('equal', adjustable='box')

    # --- 添加共享色条 ---
    fig.subplots_adjust(right=0.85, wspace=0.1, bottom=0.15)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(mappable, cax=cbar_ax, label="Temperature (K)")
    
    # --- 在图片最下方添加指标文本 ---
    metrics_text = f"Loss(MSE): {loss:.4e}  |  Relative Error: {error:.4f}"
    fig.text(0.5, 0.05, metrics_text, ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.savefig(output_filename)
    plt.close(fig)
    print(f"可视化结果已保存至: {output_filename}")


def visualize_comparison(trajectory_data: Dict[str, Any], 
                         prediction_history: np.ndarray, 
                         snapshot_indices: List[int],
                         output_dir: str):
    """循环遍历快照索引并为每个索引生成和保存可视化"""
    
    nodes = trajectory_data['nodes']
    faces = trajectory_data['faces']
    ground_truth = trajectory_data['node_features']
    time_points = trajectory_data['time_points']
    
    loss_fn = torch.nn.MSELoss()

    print("\n正在生成可视化图像...")
    for t_idx in snapshot_indices:
        gt_data = ground_truth[t_idx, :, 0]
        pred_data = prediction_history[t_idx, :, 0]
        
        gt_tensor = torch.from_numpy(gt_data)
        pred_tensor = torch.from_numpy(pred_data)
        
        loss = loss_fn(pred_tensor, gt_tensor).item()
        error = compute_state_error(pred_tensor, gt_tensor)

        time_t = time_points[t_idx]
        
        output_filename = os.path.join(output_dir, f"inference_t_{t_idx}.png")
        
        # 调用新的可视化函数，不再需要传递 vmin/vmax
        visualize_single_snapshot(nodes, gt_data, pred_data, time_t, output_filename, loss, error, faces)


def main():
    """主函数"""
    script_dir = pathlib.Path(__file__).parent.resolve()

    # --- 硬编码配置 ---
    checkpoint_path = str(script_dir / 'checkpoints' / 'best_model.pth')
    data_path = str(script_dir / 'data_laser_hardening' / 'pde_trajectories.h5')
    trajectory_key = 'trajectory_38'
    output_dir = str(script_dir / 'inference_results')
    # --------------------

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 加载模型和数据
    model = load_model_from_checkpoint(checkpoint_path, device)
    trajectory_data = get_trajectory_data(data_path, trajectory_key)

    # 2. 准备数据和几何预处理
    nodes = torch.from_numpy(trajectory_data['nodes']).float().to(device)
    edges = torch.from_numpy(trajectory_data['edges']).long().to(device)
    faces = torch.from_numpy(trajectory_data['faces']).long().to(device)
    source_terms = torch.from_numpy(trajectory_data['source_terms']).float().to(device)
    initial_condition = torch.from_numpy(trajectory_data['initial_condition']).float().to(device)
    true_trajectory = torch.from_numpy(trajectory_data['node_features']).float().to(device)
    time_points = torch.from_numpy(trajectory_data['time_points']).float().to(device)
    boundary_info = {k: {k2: v2.to(device) for k2, v2 in v.items()} for k, v in trajectory_data['boundary_info'].items()}
    
    # GKSGraph is used to conveniently calculate geometric properties
    temp_gks_graph = GKSGraph(
        nodes=nodes, edges=edges, faces=faces,
        node_features=torch.zeros_like(source_terms),
        source_terms=torch.zeros_like(source_terms),
        initial_condition=initial_condition,
        time_points=time_points,
        boundary_info=boundary_info
    )
    node_volumes = temp_gks_graph.node_volumes
    node_type = temp_gks_graph.node_type
    
    # All old geometry pre-computation is now removed as it's handled by model.forward.

    # 3. 计算需要可视化的索引
    total_steps = len(trajectory_data['time_points']) - 1
    snapshot_indices = sorted(list(set([
        0, 1, int(total_steps * 0.25), int(total_steps * 0.5), 
        int(total_steps * 0.75), total_steps
    ])))
    
    prediction_history = np.zeros_like(trajectory_data['node_features'])
    
    print("\n开始执行完整Free-Running推理...")
    with torch.no_grad():
        inference_batch = {
            'nodes': nodes, 'edges': edges, 'faces': faces,
            'node_volumes': node_volumes, 'node_type': node_type,
            'boundary_info': boundary_info,
            'initial_conditions': initial_condition.unsqueeze(0),
            'source_terms': source_terms.unsqueeze(0),
            'time_points': time_points,
            'node_features': true_trajectory.unsqueeze(0)
        }
        
        predictions = model(inference_batch)
        prediction_history = predictions['u_final'][0].cpu().numpy()

    print("\n所有推理计算完成。")
    # 4. 可视化
    visualize_comparison(trajectory_data, prediction_history, snapshot_indices, output_dir)

if __name__ == "__main__":
    main() 