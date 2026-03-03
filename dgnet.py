"""
GKSNets系统集成模块
包含主模型GKSNet、损失函数、训练基础设施等
"""

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

# 导入其他模块
from physics import build_operator, apply_bcs_to_state
from models import OperatorCorrector, NonlinearDynamicsSolver, ResidualSolver

class LUFactorizedSolver(torch.autograd.Function):
    """
    Custom autograd function to solve Ax=b using a pre-factorized matrix A,
    while correctly propagating gradients for b.
    A is considered constant for the backward pass.
    This version uses CuPy for GPU-native solving.
    """
    @staticmethod
    def forward(ctx, A_lu, b):
        """Forward pass solves the system using the pre-factorized matrix on GPU."""
        ctx.A_lu = A_lu
        
        # Reshape for solver and convert PyTorch tensor to CuPy array via DLPack (zero-copy)
        B, N, C = b.shape
        b_reshaped = b.permute(1, 0, 2).reshape(N, B * C)
        b_cp = cupy.from_dlpack(torch.to_dlpack(b_reshaped))

        x_cp = A_lu.solve(b_cp)
        
        # Convert result from CuPy array back to PyTorch tensor via DLPack (zero-copy)
        x = torch.from_dlpack(x_cp.toDlpack())
        
        # Reshape back: [N, B*C] -> [B, N, C]
        x = x.reshape(N, B, C).permute(1, 0, 2)
        
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass computes the gradient with respect to b on GPU.
        """
        A_lu = ctx.A_lu
        
        # Reshape for solver and convert PyTorch tensor to CuPy array via DLPack (zero-copy)
        B, N, C = grad_output.shape
        grad_output_reshaped = grad_output.permute(1, 0, 2).reshape(N, B * C)
        grad_output_cp = cupy.from_dlpack(torch.to_dlpack(grad_output_reshaped))

        grad_b_cp = A_lu.solve(grad_output_cp, trans='T')

        # Convert result from CuPy array back to PyTorch tensor via DLPack (zero-copy)
        grad_b = torch.from_dlpack(grad_b_cp.toDlpack())
        
        # Reshape back: [N, B*C] -> [B, N, C]
        grad_b = grad_b.reshape(N, B, C).permute(1, 0, 2)
        
        return None, grad_b


class AverageMeter:
    """
    辅助类：计算和存储平均值
    
    在一个epoch中，跨越多个batch计算和存储平均值
    """
    def __init__(self):
        self.reset()

    # 重置方法
    def reset(self):
        self.val = 0  # 当前值
        self.avg = 0  # 平均值
        self.sum = 0  # 总和
        self.count = 0  # 计数

    # 更新方法  
    def update(self, val, n=1):
        self.val = val  # 更新当前值
        self.sum += val * n  # 更新总和
        self.count += n  # 更新计数
        self.avg = self.sum / self.count  # 更新平均值

def compute_state_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """计算预测与目标之间的相对误差"""
    with torch.no_grad():
        error = torch.norm(pred - target) / torch.norm(target)
        return error.item()

class DGNet(nn.Module):
    """
    Green Kernel Superposition Networks主模型
    
    实现基于IMEX (Crank-Nicolson)格式的双路径架构：u_final = u_green + u_net
    - 物理路径u_green：通过求解线性系统 Au_phys = b 得到
    - 数据路径u_net：纯数据驱动的残差学习
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化GKSNet模型
        
        Args:
            config: 配置字典，包含所有模型参数
        """
        super().__init__()
        
        # 配置参数获取
        self.config = config
        self.spatial_dim = config['spatial_dim']
        self.feature_dim = config['feature_dim']
        self.output_dim = config['output_dim']
        self.operator_type = config.get('operator_type', 'laplace')
        self.rank = config.get('rank', 0)
        
        # 位置A：算子修正器
        self.operator_corrector = OperatorCorrector(
            spatial_dim=self.spatial_dim,
            hidden_dim=config.get('operator_hidden_dim', 64),
            num_layers=config.get('operator_num_layers', 3)
        )
        
        # 新增：非线性动力学求解器
        self.nonlinear_solver = NonlinearDynamicsSolver(
            spatial_dim=self.spatial_dim,
            node_feature_dim=self.feature_dim,
            output_dim=self.output_dim,
            hidden_dim=config.get('residual_hidden_dim', 128), # 复用residual的配置
            num_processing_layers=config.get('residual_num_layers', 5)
        )
        
        # 数据路径：残差求解器
        self.residual_solver = ResidualSolver(
            spatial_dim=self.spatial_dim,
            node_feature_dim=self.feature_dim,
            output_dim=self.output_dim,
            hidden_dim=config.get('residual_hidden_dim', 128),
            num_processing_layers=config.get('residual_num_layers', 5)
        )
    
    def forward(self, batch: Dict[str, Any], use_physics_path: bool = True, use_physics_operator: bool = True, use_nn_correction: bool = True) -> Dict[str, torch.Tensor]:
        """
        GKSNet前向传播 (IMEX版本)。

        此函数在一个批次(batch)上执行时间序列的自回归演化（Free Running）。
        它假设批次内所有样本共享相同的几何、边界条件和时间步。
        
        Args:
            batch (Dict[str, torch.Tensor]): 包含批次数据的字典
        
        Returns:
            Dict[str, torch.Tensor]: 包含模型预测结果的字典
        """
        
        # 1. 解析批次数据
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

        # 2. 初始化：预计算静态算子 (Algorithm 1, lines 1-5)
        if use_nn_correction:
            corrector_data = {'nodes': nodes, 'edges': edges, 'node_volumes': node_volumes, 'node_type': node_type}
            delta_L = self.operator_corrector(corrector_data)
        else:
            delta_L = torch.zeros((N, N), device=device)
        
        if use_physics_operator:
            L_base = build_operator(nodes=nodes, edges=edges, faces=faces, node_volumes=node_volumes, operator_type=self.operator_type)
            L_final = L_base + delta_L
        else:
            # 消融实验：仅使用学习到的算子修正
            L_final = delta_L
        
        I = torch.eye(N, device=device)
        
        # 关键修复：先计算需要梯度的B_op，再创建不需要梯度的A用于LU分解
        B_op = I + (dt / 2) * L_final
        A = I - (dt / 2) * L_final.detach()
        
        # 预计算A的稀疏LU分解 (迁移到GPU上)
        A_torch_dlpack = torch.to_dlpack(A)
        A_cupy = cupy.from_dlpack(A_torch_dlpack)
        A_cupy_sparse = cupy.sparse.csc_matrix(A_cupy)
        A_lu = cspl.splu(A_cupy_sparse)

        # 3. 初始化轨迹和当前状态 (Algorithm 1, line 6)
        u_final_history = torch.zeros(B, T, N, self.output_dim, device=device)
        u_current = initial_conditions
        u_final_history[:, 0] = u_current
        
        # 4. 时间演化循环 (Algorithm 1, line 7-16)
        for t in range(T - 1):
            # 获取当前和下一时间步的源项
            f_current = source_terms[:, t]
            f_next = source_terms[:, t + 1]
            
            # --- 物理路径 ---
            if use_physics_path:
                # a. GNN学习非线性源项 (Algorithm 1, line 8)
                r_uk_batch = torch.zeros_like(u_current)
                for b in range(B):
                    nonlinear_data = {
                        'nodes': nodes, 'edges': edges,
                        'node_features': u_current[b], 'node_type': node_type
                    }
                    r_uk_batch[b] = self.nonlinear_solver(nonlinear_data)

                # b. 构建右手侧向量 b (Algorithm 1, lines 9-11)
                b_linear = B_op @ u_current
                b_source = (dt / 2) * (f_current + f_next)
                b = b_linear + b_source + dt * r_uk_batch
                
                # c. 求解线性系统 Au_phys = b (Algorithm 1, line 12)
                # 使用自定义的autograd函数进行求解
                u_phys_next = LUFactorizedSolver.apply(A_lu, b)
            else:
                # 如果禁用物理路径，则物理部分的解为零
                u_phys_next = torch.zeros_like(u_current)

            # --- 数据路径 (残差修正) --- (Algorithm 1, line 13)
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
            
            # --- 最终叠加 --- (Algorithm 1, line 14)
            u_final_current = u_phys_next + u_net_next

            # 应用边界条件
            if boundary_info:
                u_final_current = apply_bcs_to_state(u_final_current.clone().squeeze(-1), boundary_info).unsqueeze(-1)
            
            # 更新历史和当前状态 (Algorithm 1, line 15)
            u_final_history[:, t + 1] = u_final_current
            u_current = u_final_current.detach()

        # Teacher-Forcing 模式被移除，只保留Free-Running
        
        return {
            'u_final': u_final_history
            # 'u_green' and 'u_net' are intermediate, not returned for simplicity
        }

class Loss(nn.Module):
    """
    GKSNet专用损失函数：只监督最终预测的首末时间步。
    
    实现Free Running模式下的首末时间步损失：
    Loss = MSE(u_final_pred^(1), u_true^(1)) + MSE(u_final_pred^(T-1), u_true^(T-1))
    
    该损失函数只关心融合后的最终预测结果，旨在确保单步精度和长期稳定性。
    """
    
    def __init__(self, config: Dict):
        """
        初始化损失函数
        
        Args:
            config: 配置字典，可以包含：
                - 'loss_type': 损失类型，可选 'mse', 'mae', 'huber'，默认为 'mse'。
        """
        super().__init__()
        
        # 获取损失类型
        self.loss_type = config.get('loss_type', 'mse')
        
        # 定义基础损失函数
        if self.loss_type == 'mse':
            self.base_loss = nn.MSELoss()
        elif self.loss_type == 'mae':
            self.base_loss = nn.L1Loss()
        elif self.loss_type == 'huber':
            self.base_loss = nn.HuberLoss()
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算首末时间步损失
        
        Args:
            predictions: 预测结果字典，必须包含：
                - 'u_final': [B, T, N, output_dim] 最终预测，B=批次大小，T=时间步数，N=节点数，output_dim=输出维度
            targets: [B, T, N, output_dim] 真实目标值
            
        Returns:
            Dict[str, torch.Tensor]: 包含总损失和各部分损失的字典
        """
        
        # 提取最终预测结果
        u_final_pred = predictions['u_final']  # [B, T, N, output_dim]
        
        # 首时间步损失 (t=1)
        first_step_loss = self.base_loss(u_final_pred[:, 1], targets[:, 1])
        
        # 末时间步损失 (t=T-1)
        final_step_loss = self.base_loss(u_final_pred[:, -1], targets[:, -1])
        
        # 总损失
        total_loss = first_step_loss + final_step_loss
        
        return {
            'total_loss': total_loss,
            'first_step_loss': first_step_loss,
            'final_step_loss': final_step_loss,
        }

class DGTrainer:
    """
    GKSNets的训练器
    
    封装完整的训练、验证和测试流程
    """
    
    def __init__(self, 
                 model: DGNet,
                 optimizer: optim.Optimizer,
                 loss_fn: Loss,
                 config: Dict[str, Any],
                 rank: int,  # DDP rank
                 local_rank: int, # DDP local_rank
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None):
        """
        初始化训练器
        
        Args:
            model: GKSNet模型
            optimizer: 优化器
            loss_fn: 损失函数
            config: 配置字典
            rank: DDP全局进程ID
            local_rank: DDP本地GPU ID
            scheduler: 学习率调度器（可选）
        """
        # 模型、优化器、损失函数、配置
        self.model_device = torch.device(local_rank)
        self.rank = rank
        self.local_rank = local_rank
        self.config = config
        
        # 将模型移动到指定的GPU
        model.to(self.model_device)
        self.model = DDP(model, device_ids=[self.local_rank])
        
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
        
        # 持久化的指标跟踪器，避免重复创建对象
        self.train_loss_meter = AverageMeter()
        self.train_first_step_loss_meter = AverageMeter()
        self.train_final_step_loss_meter = AverageMeter()
        self.train_error_meter = AverageMeter()

        # 验证指标跟踪器
        self.val_loss_meter = AverageMeter()
        self.val_first_step_loss_meter = AverageMeter()
        self.val_final_step_loss_meter = AverageMeter()
        self.val_error_meter = AverageMeter()
        
        # 检查点目录：确保总是保存在gksNet/checkpoints/下
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_checkpoint_dir = os.path.join(script_dir, 'checkpoints')
        self.checkpoint_dir = config.get('checkpoint_dir', default_checkpoint_dir)
        
        # 只有主进程才创建目录
        if self.rank == 0:
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int):
        """
        训练流程的总入口
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器  
            num_epochs: 训练轮数
        """
        if self.rank == 0:
            print(f"Starting training for {num_epochs} epochs...")
            print(f"Device: cuda, World Size: {dist.get_world_size()}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # 设置sampler的epoch，确保每个epoch的数据洗牌不同
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            
            # 训练一个epoch
            train_metrics = self._train_epoch(train_loader)
            
            # 验证一个epoch
            val_metrics = self._evaluate_epoch(val_loader)
            
            # 只有主进程记录和打印
            if self.rank == 0:
                # 记录历史
                for key, value in train_metrics.items():
                    self.train_history[key].append(value)
                for key, value in val_metrics.items():
                    self.val_history[key].append(value)
                
                # 打印进度
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Train - Loss: {train_metrics['loss']:.6f}, "
                      f"FirstStep: {train_metrics['first_step_loss']:.6f}, "
                      f"FinalStep: {train_metrics['final_step_loss']:.6f}, "
                      f"Error: {train_metrics['relative_error']:.6f}")
                print(f"  Val   - Loss: {val_metrics['loss']:.6f}, "
                      f"FirstStep: {val_metrics['first_step_loss']:.6f}, "
                      f"FinalStep: {val_metrics['final_step_loss']:.6f}, "
                      f"Error: {val_metrics['relative_error']:.6f}")
                
                # 保存最佳模型
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint('best_model.pth')
                    print(f"  New best model saved!")

                # 保存最新模型
                self.save_checkpoint('new_model.pth')
                print(f"  Saved latest model for epoch {epoch+1} to new_model.pth")
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()

            if self.rank == 0:
                print("-" * 50)
    
    def _train_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        执行一个完整的训练epoch
        
        Args:
            data_loader: 训练数据加载器
            
        Returns:
            Dict: 训练指标
        """
        # 设置模型为训练模式
        self.model.train()
        
        # 重置指标跟踪器
        self.train_loss_meter.reset()
        self.train_first_step_loss_meter.reset()
        self.train_final_step_loss_meter.reset()
        self.train_error_meter.reset()
        
        # 仅在主进程显示tqdm进度条
        progress_bar = tqdm(data_loader, desc=f"Epoch {self.current_epoch + 1} Training", unit="batch", disable=(self.rank != 0))

        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            batch = self._move_batch_to_device(batch)
            
            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(batch)
            
            # 计算损失
            targets = batch['node_features']  # [B, T, N, feature_dim] 完整时间序列作为目标
            loss_dict = self.loss_fn(predictions, targets)
            
            # 反向传播
            loss_dict['total_loss'].backward()
            
            # 梯度裁剪（可选）
            if self.config.get('gradient_clip', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clip']
                )
            
            # 更新模型参数
            self.optimizer.step()
            
            # 更新指标
            self.train_loss_meter.update(loss_dict['total_loss'].item())
            self.train_first_step_loss_meter.update(loss_dict['first_step_loss'].item())
            self.train_final_step_loss_meter.update(loss_dict['final_step_loss'].item())
            
            # 计算预测误差
            with torch.no_grad():
                error = compute_state_error(predictions['u_final'][:, -1], targets[:, -1])
                self.train_error_meter.update(error)
            
            # 只在主进程更新tqdm描述
            if self.rank == 0:
                progress_bar.set_postfix({
                    'Loss': self.train_loss_meter.avg,
                    'Error': self.train_error_meter.avg
                })

        # --- DDP 指标同步 ---
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
        """
        执行一个完整的验证epoch
        
        Args:
            data_loader: 验证数据加载器
            
        Returns:
            Dict: 验证指标
        """
        self.model.eval()
        
        # 重置验证指标跟踪器
        self.val_loss_meter.reset()
        self.val_first_step_loss_meter.reset()
        self.val_final_step_loss_meter.reset()
        self.val_error_meter.reset()
        
        # 验证一个epoch
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc=f"Epoch {self.current_epoch + 1} Validation", unit="batch", disable=(self.rank != 0))
            for batch in progress_bar:
                # 移动数据到设备
                batch = self._move_batch_to_device(batch)
                
                # 前向传播
                predictions = self.model(batch)
                
                # 计算损失
                targets = batch['node_features']  # [B, T, N, feature_dim] 完整时间序列作为目标
                loss_dict = self.loss_fn(predictions, targets)
                
                # 更新指标
                self.val_loss_meter.update(loss_dict['total_loss'].item())
                self.val_first_step_loss_meter.update(loss_dict['first_step_loss'].item())
                self.val_final_step_loss_meter.update(loss_dict['final_step_loss'].item())
                
                # 计算预测误差
                error = compute_state_error(predictions['u_final'][:, -1], targets[:, -1])
                self.val_error_meter.update(error)

        # --- DDP 指标同步 ---
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
        """递归地将数据结构中的所有张量移动到指定设备"""
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
        """
        保存模型检查点 (仅在主进程执行)
        
        Args:
            filename: 保存文件名
        """
        if self.rank != 0:
            return

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.module.state_dict(), # 保存 unwrapped model
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
        """
        加载模型检查点 (所有进程都需要加载以同步模型)
        
        Args:
            filename: 检查点文件名
        """
        load_path = os.path.join(self.checkpoint_dir, filename)
        
        # 使用 map_location 将模型加载到每个进程对应的GPU上
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

