"""
GKSNet简单训练脚本 - 用于验证代码架构
"""

import os
import torch
import torch.optim as optim
import pathlib
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import h5py

# 导入GKSNet模块
from gks_net import GKSNet, Loss, GKSTrainer
from dataset import GKSPdeDataset, create_gks_loader

def setup_ddp():
    """初始化DDP环境"""
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    """清理DDP环境"""
    dist.destroy_process_group()

def main():
    """主函数"""
    
    # 初始化DDP
    setup_ddp()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # 数据目录设置
    base_dir = pathlib.Path(__file__).parent.resolve()
    data_path_absolute = base_dir / 'data_laser_hardening' / 'pde_trajectories.h5'

    # 简单配置
    config = {
        # 数据配置
        'data_path': data_path_absolute,
        'batch_size': 4,
        'num_workers': 2,
        'train_time_steps': 7, # 定义训练时使用的子轨迹时间步长度 M
        
        # 模型配置
        'spatial_dim': 2,
        'feature_dim': 1,
        'output_dim': 1,
        'operator_type': 'laplace',
        
        # 网络结构配置
        'operator_hidden_dim': 64,
        'operator_num_layers': 3,
        'residual_hidden_dim': 128,
        'residual_num_layers': 5,
        
        # 训练配置
        'num_epochs': 40,
        'learning_rate': 5e-4,
        'lr_decay_step_size': 20,
        'lr_decay_gamma': 0.1,
        
        # DDP rank
        'rank': rank,
        
        # 设备配置 (DDP中不再需要device配置，直接使用local_rank)
        'save_dir': str(base_dir / 'checkpoints'),
        'checkpoint_dir': str(base_dir / 'checkpoints'),
        'log_interval': 1,
    }
    
    # 只有主进程打印信息
    if rank == 0:
        print("GKSNet架构验证训练 (DDP模式)")
        print(f"检测到 {world_size} 个GPUs.")
        print(f"数据路径: {config['data_path']}")
    
    # 检查数据文件
    if not os.path.exists(config['data_path']):
        # 只让主进程打印错误并退出，防止多进程重复打印
        if rank == 0:
            print(f"错误: 数据文件不存在 - {config['data_path']}")
            print("请确保数据文件存在后再运行")
        return
    
    # 只有主进程创建保存目录
    if rank == 0:
        os.makedirs(config['save_dir'], exist_ok=True)
    
    # 加载数据
    if rank == 0:
        print("\n1. 加载数据...")
    # 按轨迹划分训练/验证，完全隔离，避免数据泄露
    with h5py.File(config['data_path'], 'r') as f:
        all_traj_keys = sorted(list(f.keys()))
    total_traj = len(all_traj_keys)
    if total_traj < 2:
        raise ValueError("轨迹数量不足，无法划分训练/验证集")
    split_idx = max(1, int(0.8 * total_traj))
    train_traj_keys = all_traj_keys[:split_idx]
    val_traj_keys = all_traj_keys[split_idx:]
    if len(val_traj_keys) == 0:
        val_traj_keys = [train_traj_keys.pop()]
    if rank == 0:
        print(f"   轨迹总数: {total_traj}")
        print(f"   训练轨迹数: {len(train_traj_keys)}")
        print(f"   验证轨迹数: {len(val_traj_keys)}")

    train_dataset = GKSPdeDataset(
        data_path=config['data_path'],
        train_time_steps=config['train_time_steps'],
        rank=rank,
        trajectory_keys=train_traj_keys
    )
    val_dataset = GKSPdeDataset(
        data_path=config['data_path'],
        train_time_steps=config['train_time_steps'],
        rank=rank,
        trajectory_keys=val_traj_keys
    )
    if rank == 0:
        print(f"   训练样本数: {len(train_dataset)}")
        print(f"   验证样本数: {len(val_dataset)}")
    
    # 创建分布式采样器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    # 创建数据加载器
    train_loader = create_gks_loader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=False,  # shuffle由sampler控制
        num_workers=config['num_workers'],
        sampler=train_sampler
    )
    val_loader = create_gks_loader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,  # shuffle由sampler控制
        num_workers=config['num_workers'],
        sampler=val_sampler
    )
    
    # 创建模型
    if rank == 0:
        print("\n2. 创建模型...")
    model = GKSNet(config)
    if rank == 0:
        print(f"   模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_fn = Loss(config)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config['lr_decay_step_size'], 
        gamma=config['lr_decay_gamma']
    )
    
    # 创建训练器
    trainer = GKSTrainer(model, optimizer, loss_fn, config, rank=rank, local_rank=local_rank, scheduler=scheduler)
    
    # 开始训练
    if rank == 0:
        print("\n3. 开始训练...")
    trainer.train(train_loader, val_loader, config['num_epochs'])
    
    if rank == 0:
        print("\n训练完成！架构验证成功。")

    # 清理DDP
    cleanup_ddp()


if __name__ == "__main__":
    main() 