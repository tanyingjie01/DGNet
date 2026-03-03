"""DGNet inference and visualization script."""

import os
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

from dgnet import DGNet
from dataset import DGGraph

def compute_state_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute relative state error."""
    with torch.no_grad():
        pred = pred.float()
        target = target.float()
        target_norm = torch.norm(target)
        if target_norm == 0:
            return 0.0 if torch.norm(pred) == 0 else float('inf')
        error = torch.norm(pred - target) / target_norm
        return error.item()

def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> DGNet:
    """Load DGNet from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading model from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = DGNet(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print("Model loaded successfully.")
    return model

def get_trajectory_data(data_path: str, trajectory_key: str) -> Dict[str, np.ndarray]:
    """Load one trajectory from HDF5."""
    print(f"Loading trajectory '{trajectory_key}' from '{data_path}'...")
    with h5py.File(data_path, 'r') as f:
        if trajectory_key not in f:
            raise KeyError(f"Trajectory not found in HDF5 file: {trajectory_key}")
        
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
        
    print("Trajectory data loaded.")
    return data

def visualize_single_snapshot(nodes, ground_truth_t, prediction_t, time_point_t, output_filename, loss, error, faces):
    """Render and save one side-by-side snapshot."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), squeeze=False)
    fig.suptitle(f"Inference at t={time_point_t:.3f}s", fontsize=16)

    triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], triangles=faces)

    T_ambient = 298.15

    combined_data = np.concatenate([ground_truth_t, prediction_t])
    heated_data = combined_data[combined_data > T_ambient + 1.0]
    if heated_data.size > 0:
        vmax = np.percentile(heated_data, 99.9)
    else:
        vmax = T_ambient + 1.0

    vmin = T_ambient
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    ax_gt = axes[0, 0]
    mappable = ax_gt.tripcolor(triangulation, ground_truth_t, cmap='hot', norm=norm, shading='gouraud')
    ax_gt.set_title("Ground Truth")
    ax_gt.set_xlabel("x")
    ax_gt.set_ylabel("y")
    ax_gt.set_aspect('equal', adjustable='box')

    ax_pred = axes[0, 1]
    ax_pred.tripcolor(triangulation, prediction_t, cmap='hot', norm=norm, shading='gouraud')
    ax_pred.set_title("Prediction")
    ax_pred.set_xlabel("x")
    ax_pred.set_ylabel("")
    ax_pred.set_yticklabels([])
    ax_pred.set_aspect('equal', adjustable='box')

    fig.subplots_adjust(right=0.85, wspace=0.1, bottom=0.15)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(mappable, cax=cbar_ax, label="Temperature (K)")
    
    metrics_text = f"MSE: {loss:.4e}  |  Relative Error: {error:.4f}"
    fig.text(0.5, 0.05, metrics_text, ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.savefig(output_filename)
    plt.close(fig)


def visualize_comparison(trajectory_data: Dict[str, Any], 
                         prediction_history: np.ndarray, 
                         snapshot_indices: List[int],
                         output_dir: str):
    """Generate plots for selected time indices."""
    
    nodes = trajectory_data['nodes']
    faces = trajectory_data['faces']
    ground_truth = trajectory_data['node_features']
    time_points = trajectory_data['time_points']
    
    loss_fn = torch.nn.MSELoss()

    print("\nGenerating visualizations...")
    for t_idx in snapshot_indices:
        gt_data = ground_truth[t_idx, :, 0]
        pred_data = prediction_history[t_idx, :, 0]
        
        gt_tensor = torch.from_numpy(gt_data)
        pred_tensor = torch.from_numpy(pred_data)
        
        loss = loss_fn(pred_tensor, gt_tensor).item()
        error = compute_state_error(pred_tensor, gt_tensor)

        time_t = time_points[t_idx]
        time_seconds = int(round(float(time_t)))
        output_filename = os.path.join(output_dir, f"inference_t_{time_seconds}.png")
        
        visualize_single_snapshot(nodes, gt_data, pred_data, time_t, output_filename, loss, error, faces)

    print("Visualizations saved.")


def main():
    """Run inference and export figures."""
    base_dir = pathlib.Path(__file__).parent.resolve()

    checkpoint_path = str(base_dir / 'checkpoints' / 'best_model.pth')
    data_path = str(base_dir / 'data_laser_hardening' / 'pde_trajectories.h5')
    # trajectory_39 is in the validation split used by train.py under lexicographic ordering.
    # Validation set there is: ['trajectory_4', 'trajectory_5', 'trajectory_6', 'trajectory_7', 'trajectory_8', 'trajectory_9', 'trajectory_38', 'trajectory_39']
    trajectory_key = 'trajectory_39'
    output_dir = str(base_dir / 'inference_results')

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model_from_checkpoint(checkpoint_path, device)
    trajectory_data = get_trajectory_data(data_path, trajectory_key)

    nodes = torch.from_numpy(trajectory_data['nodes']).float().to(device)
    edges = torch.from_numpy(trajectory_data['edges']).long().to(device)
    faces = torch.from_numpy(trajectory_data['faces']).long().to(device)
    source_terms = torch.from_numpy(trajectory_data['source_terms']).float().to(device)
    initial_condition = torch.from_numpy(trajectory_data['initial_condition']).float().to(device)
    true_trajectory = torch.from_numpy(trajectory_data['node_features']).float().to(device)
    time_points = torch.from_numpy(trajectory_data['time_points']).float().to(device)
    boundary_info = {k: {k2: v2.to(device) for k2, v2 in v.items()} for k, v in trajectory_data['boundary_info'].items()}
    
    temp_dg_graph = DGGraph(
        nodes=nodes, edges=edges, faces=faces,
        node_features=torch.zeros_like(source_terms),
        source_terms=torch.zeros_like(source_terms),
        initial_condition=initial_condition,
        time_points=time_points,
        boundary_info=boundary_info
    )
    node_volumes = temp_dg_graph.node_volumes
    node_type = temp_dg_graph.node_type
    
    snapshot_indices = [0, 30, 60, 90, 120]
    
    prediction_history = np.zeros_like(trajectory_data['node_features'])
    
    print("\nRunning full free-running inference...")
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

    print("\nAll inference computations completed.")
    final_gt = true_trajectory[-1, :, 0].cpu()
    final_pred = torch.from_numpy(prediction_history[-1, :, 0])
    final_mse = torch.nn.functional.mse_loss(final_pred, final_gt).item()
    final_rne = compute_state_error(final_pred, final_gt)
    print(f"Final-step metrics - MSE: {final_mse:.4e}  |  RNE: {final_rne:.4f}")
    visualize_comparison(trajectory_data, prediction_history, snapshot_indices, output_dir)

if __name__ == "__main__":
    main() 