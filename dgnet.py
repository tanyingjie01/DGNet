"""DGNet model, loss, and training utilities."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import time
import os
from collections import defaultdict
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import cupy
import cupy.sparse.linalg as cspl

from physics import build_operator, apply_bcs_to_state
from models import OperatorCorrector, NonlinearDynamicsSolver, ResidualSolver

class LUFactorizedSolver(torch.autograd.Function):
    """Solve Ax=b with a pre-factorized CuPy LU handle."""
    @staticmethod
    def forward(ctx, A_lu, b):
        """Forward solve."""
        ctx.A_lu = A_lu

        B, N, C = b.shape
        b_reshaped = b.permute(1, 0, 2).reshape(N, B * C)
        b_cp = cupy.from_dlpack(torch.to_dlpack(b_reshaped))

        x_cp = A_lu.solve(b_cp)

        x = torch.from_dlpack(x_cp.toDlpack())

        x = x.reshape(N, B, C).permute(1, 0, 2)

        return x

    @staticmethod
    def backward(ctx, grad_output):
        """Backward solve for grad_b."""
        A_lu = ctx.A_lu

        B, N, C = grad_output.shape
        grad_output_reshaped = grad_output.permute(1, 0, 2).reshape(N, B * C)
        grad_output_cp = cupy.from_dlpack(torch.to_dlpack(grad_output_reshaped))

        grad_b_cp = A_lu.solve(grad_output_cp, trans='T')

        grad_b = torch.from_dlpack(grad_b_cp.toDlpack())

        grad_b = grad_b.reshape(N, B, C).permute(1, 0, 2)

        return None, grad_b


class AverageMeter:
    """Track running averages."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_state_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute relative error."""
    with torch.no_grad():
        error = torch.norm(pred - target) / torch.norm(target)
        return error.item()

class DGNet(nn.Module):
    """Discrete Green Network with IMEX time stepping."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize DGNet."""
        super().__init__()

        self.config = config
        self.spatial_dim = config['spatial_dim']
        self.feature_dim = config['feature_dim']
        self.output_dim = config['output_dim']
        self.operator_type = config.get('operator_type', 'laplace')
        self.rank = config.get('rank', 0)
        
        self.operator_corrector = OperatorCorrector(
            spatial_dim=self.spatial_dim,
            hidden_dim=config.get('operator_hidden_dim', 64),
            num_layers=config.get('operator_num_layers', 3)
        )
        
        self.nonlinear_solver = NonlinearDynamicsSolver(
            spatial_dim=self.spatial_dim,
            node_feature_dim=self.feature_dim,
            output_dim=self.output_dim,
            hidden_dim=config.get('residual_hidden_dim', 128),
            num_processing_layers=config.get('residual_num_layers', 5)
        )

        self.residual_solver = ResidualSolver(
            spatial_dim=self.spatial_dim,
            node_feature_dim=self.feature_dim,
            output_dim=self.output_dim,
            hidden_dim=config.get('residual_hidden_dim', 128),
            num_processing_layers=config.get('residual_num_layers', 5)
        )
    
    def forward(self, batch: Dict[str, Any], use_physics_path: bool = True, use_physics_operator: bool = True, use_nn_correction: bool = True) -> Dict[str, torch.Tensor]:
        """Run free-running rollout on one batch."""

        nodes = batch['nodes']
        edges = batch['edges']
        faces = batch['faces']
        node_volumes = batch['node_volumes']
        initial_conditions = batch['initial_conditions']
        source_terms = batch['source_terms']
        time_points = batch['time_points']
        node_type = batch.get('node_type', None)
        boundary_info = batch['boundary_info']
        
        B, T, N, _ = source_terms.shape
        device = nodes.device
        dt = float(time_points[1] - time_points[0]) if T > 1 else 0.0

        if use_nn_correction:
            corrector_data = {'nodes': nodes, 'edges': edges, 'node_volumes': node_volumes, 'node_type': node_type}
            delta_L = self.operator_corrector(corrector_data)
        else:
            delta_L = torch.zeros((N, N), device=device)
        
        if use_physics_operator:
            L_base = build_operator(nodes=nodes, edges=edges, faces=faces, node_volumes=node_volumes, operator_type=self.operator_type)
            L_final = L_base + delta_L
        else:
            L_final = delta_L
        
        I = torch.eye(N, device=device)
        
        B_op = I + (dt / 2) * L_final
        A = I - (dt / 2) * L_final.detach()

        A_torch_dlpack = torch.to_dlpack(A)
        A_cupy = cupy.from_dlpack(A_torch_dlpack)
        A_cupy_sparse = cupy.sparse.csc_matrix(A_cupy)
        A_lu = cspl.splu(A_cupy_sparse)

        u_final_history = torch.zeros(B, T, N, self.output_dim, device=device)
        u_current = initial_conditions
        u_final_history[:, 0] = u_current
        
        for t in range(T - 1):
            f_current = source_terms[:, t]
            f_next = source_terms[:, t + 1]

            if use_physics_path:
                r_uk_batch = torch.zeros_like(u_current)
                for b in range(B):
                    nonlinear_data = {
                        'nodes': nodes, 'edges': edges,
                        'node_features': u_current[b], 'node_type': node_type
                    }
                    r_uk_batch[b] = self.nonlinear_solver(nonlinear_data)

                b_linear = B_op @ u_current
                b_source = (dt / 2) * (f_current + f_next)
                b = b_linear + b_source + dt * r_uk_batch

                u_phys_next = LUFactorizedSolver.apply(A_lu, b)
            else:
                u_phys_next = torch.zeros_like(u_current)

            if use_nn_correction:
                u_net_next = torch.zeros_like(u_current)
                for b in range(B):
                    residual_data = {
                        'nodes': nodes, 'edges': edges,
                        'node_features': u_current[b],
                        'boundary_info': boundary_info, 'node_type': node_type
                    }
                    u_net_next[b] = self.residual_solver(residual_data)
            else:
                u_net_next = torch.zeros_like(u_current)
            
            u_final_current = u_phys_next + u_net_next

            if boundary_info:
                u_final_current = apply_bcs_to_state(u_final_current.clone().squeeze(-1), boundary_info).unsqueeze(-1)

            u_final_history[:, t + 1] = u_final_current
            u_current = u_final_current.detach()

        return {
            'u_final': u_final_history
        }

class Loss(nn.Module):
    """Loss on first and last predicted steps."""
    
    def __init__(self, config: Dict):
        """Initialize loss function."""
        super().__init__()

        self.loss_type = config.get('loss_type', 'mse')

        if self.loss_type == 'mse':
            self.base_loss = nn.MSELoss()
        elif self.loss_type == 'mae':
            self.base_loss = nn.L1Loss()
        elif self.loss_type == 'huber':
            self.base_loss = nn.HuberLoss()
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute loss dictionary."""

        u_final_pred = predictions['u_final']

        first_step_loss = self.base_loss(u_final_pred[:, 1], targets[:, 1])

        final_step_loss = self.base_loss(u_final_pred[:, -1], targets[:, -1])

        total_loss = first_step_loss + final_step_loss
        
        return {
            'total_loss': total_loss,
            'first_step_loss': first_step_loss,
            'final_step_loss': final_step_loss,
        }

class DGTrainer:
    """Training and validation loop for DGNet."""
    
    def __init__(self, 
                 model: DGNet,
                 optimizer: optim.Optimizer,
                 loss_fn: Loss,
                 config: Dict[str, Any],
                 rank: int,
                 local_rank: int,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None):
        """Initialize trainer state."""
        self.model_device = torch.device(local_rank)
        self.rank = rank
        self.local_rank = local_rank
        self.config = config
        
        model.to(self.model_device)
        self.model = DDP(model, device_ids=[self.local_rank])
        
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
        
        self.train_loss_meter = AverageMeter()
        self.train_first_step_loss_meter = AverageMeter()
        self.train_final_step_loss_meter = AverageMeter()
        self.train_error_meter = AverageMeter()

        self.val_loss_meter = AverageMeter()
        self.val_first_step_loss_meter = AverageMeter()
        self.val_final_step_loss_meter = AverageMeter()
        self.val_error_meter = AverageMeter()
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_checkpoint_dir = os.path.join(script_dir, 'checkpoints')
        self.checkpoint_dir = config.get('checkpoint_dir', default_checkpoint_dir)
        
        if self.rank == 0:
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int):
        """Run training for a fixed number of epochs."""
        if self.rank == 0:
            print(f"Starting training for {num_epochs} epochs...")
            print(f"Device: cuda, World Size: {dist.get_world_size()}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            train_metrics = self._train_epoch(train_loader)

            val_metrics = self._evaluate_epoch(val_loader)

            if self.rank == 0:
                for key, value in train_metrics.items():
                    self.train_history[key].append(value)
                for key, value in val_metrics.items():
                    self.val_history[key].append(value)

                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Train - Loss: {train_metrics['loss']:.6f}, "
                      f"FirstStep: {train_metrics['first_step_loss']:.6f}, "
                      f"FinalStep: {train_metrics['final_step_loss']:.6f}, "
                      f"Error: {train_metrics['relative_error']:.6f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.6f}, "
                      f"FirstStep: {val_metrics['first_step_loss']:.6f}, "
                      f"FinalStep: {val_metrics['final_step_loss']:.6f}, "
                      f"Error: {val_metrics['relative_error']:.6f}")
                
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model.pth')
                    print(f"  New best model saved!")

                self.save_checkpoint('new_model.pth')
                print(f"  Saved latest model for epoch {epoch+1} to new_model.pth")

            if self.scheduler:
                self.scheduler.step()

            if self.rank == 0:
                print("-" * 50)
    
    def _train_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()

        self.train_loss_meter.reset()
        self.train_first_step_loss_meter.reset()
        self.train_final_step_loss_meter.reset()
        self.train_error_meter.reset()
        
        progress_bar = tqdm(data_loader, desc=f"Epoch {self.current_epoch + 1} Training", unit="batch", disable=(self.rank != 0))

        for batch_idx, batch in enumerate(progress_bar):
            batch = self._move_batch_to_device(batch)

            self.optimizer.zero_grad()
            predictions = self.model(batch)

            targets = batch['node_features']
            loss_dict = self.loss_fn(predictions, targets)

            loss_dict['total_loss'].backward()

            if self.config.get('gradient_clip', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()

            self.train_loss_meter.update(loss_dict['total_loss'].item())
            self.train_first_step_loss_meter.update(loss_dict['first_step_loss'].item())
            self.train_final_step_loss_meter.update(loss_dict['final_step_loss'].item())
            
            with torch.no_grad():
                error = compute_state_error(predictions['u_final'][:, -1], targets[:, -1])
                self.train_error_meter.update(error)

            if self.rank == 0:
                progress_bar.set_postfix({
                    'Loss': self.train_loss_meter.avg,
                    'Error': self.train_error_meter.avg
                })

        metrics_tensor = torch.tensor([
            self.train_loss_meter.avg,
            self.train_first_step_loss_meter.avg,
            self.train_final_step_loss_meter.avg,
            self.train_error_meter.avg
        ], device=self.model_device)
        
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        metrics_tensor /= dist.get_world_size()
        
        return {
            'loss': metrics_tensor[0].item(),
            'first_step_loss': metrics_tensor[1].item(),
            'final_step_loss': metrics_tensor[2].item(),
            'relative_error': metrics_tensor[3].item()
        }
    
    def _evaluate_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """Run one validation epoch."""
        self.model.eval()

        self.val_loss_meter.reset()
        self.val_first_step_loss_meter.reset()
        self.val_final_step_loss_meter.reset()
        self.val_error_meter.reset()
        
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc=f"Epoch {self.current_epoch + 1} Validation", unit="batch", disable=(self.rank != 0))
            for batch in progress_bar:
                batch = self._move_batch_to_device(batch)

                predictions = self.model(batch)

                targets = batch['node_features']
                loss_dict = self.loss_fn(predictions, targets)

                self.val_loss_meter.update(loss_dict['total_loss'].item())
                self.val_first_step_loss_meter.update(loss_dict['first_step_loss'].item())
                self.val_final_step_loss_meter.update(loss_dict['final_step_loss'].item())
                
                error = compute_state_error(predictions['u_final'][:, -1], targets[:, -1])
                self.val_error_meter.update(error)

        metrics_tensor = torch.tensor([
            self.val_loss_meter.avg,
            self.val_first_step_loss_meter.avg,
            self.val_final_step_loss_meter.avg,
            self.val_error_meter.avg
        ], device=self.model_device)
        
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        metrics_tensor /= dist.get_world_size()

        return {
            'loss': metrics_tensor[0].item(),
            'first_step_loss': metrics_tensor[1].item(),
            'final_step_loss': metrics_tensor[2].item(),
            'relative_error': metrics_tensor[3].item()
        }
    
    def _move_batch_to_device(self, data: Any) -> Any:
        """Recursively move tensors to the training device."""
        if isinstance(data, torch.Tensor):
            return data.to(self.model_device)
        if isinstance(data, dict):
            return {k: self._move_batch_to_device(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._move_batch_to_device(v) for v in data]
        if isinstance(data, tuple):
            return tuple(self._move_batch_to_device(v) for v in data)
        return data
    
    def save_checkpoint(self, filename: str):
        """Save checkpoint on rank 0."""
        if self.rank != 0:
            return

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'train_history': dict(self.train_history),
            'val_history': dict(self.val_history)
        }
        
        save_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, filename: str):
        """Load checkpoint and restore trainer state."""
        load_path = os.path.join(self.checkpoint_dir, filename)
        
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.local_rank}
        checkpoint = torch.load(load_path, map_location=map_location)
        
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = defaultdict(list, checkpoint['train_history'])
        self.val_history = defaultdict(list, checkpoint['val_history'])
        
        if self.rank == 0:
            print(f"Checkpoint loaded from {load_path}")
            print(f"Resuming from epoch {self.current_epoch + 1}, best val loss: {self.best_val_loss:.6f}")

