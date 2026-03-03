"""DGNet training entrypoint (DDP)."""

import os
import torch
import torch.optim as optim
import pathlib
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import h5py

from dgnet import DGNet, Loss, DGTrainer
from dataset import DGPdeDataset, create_dg_loader

def setup_ddp():
    """Initialize DDP runtime."""
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    """Tear down DDP runtime."""
    dist.destroy_process_group()

def main():
    """Run training."""

    setup_ddp()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    base_dir = pathlib.Path(__file__).parent.resolve()
    data_path_absolute = base_dir / 'data_laser_hardening' / 'pde_trajectories.h5'

    config = {
        'data_path': data_path_absolute,
        'batch_size': 4,
        'num_workers': 2,
        'train_time_steps': 7,

        'spatial_dim': 2,
        'feature_dim': 1,
        'output_dim': 1,
        'operator_type': 'laplace',

        'operator_hidden_dim': 64,
        'operator_num_layers': 3,
        'residual_hidden_dim': 128,
        'residual_num_layers': 5,

        'num_epochs': 15,
        'learning_rate': 5e-4,
        'lr_decay_step_size': 5,
        'lr_decay_gamma': 0.2,

        'rank': rank,

        'save_dir': str(base_dir / 'checkpoints'),
        'checkpoint_dir': str(base_dir / 'checkpoints'),
        'log_interval': 1,
    }

    if rank == 0:
        print("DGNet architecture validation training (DDP)")
        print(f"Detected {world_size} GPUs.")
        print(f"Data path: {config['data_path']}")

    if not os.path.exists(config['data_path']):
        if rank == 0:
            print(f"Error: data file not found - {config['data_path']}")
            print("Please ensure the data file exists before running.")
        return

    if rank == 0:
        os.makedirs(config['save_dir'], exist_ok=True)

    if rank == 0:
        print("\nLoading data...")

    with h5py.File(config['data_path'], 'r') as f:
        # NOTE (important): keys are sorted lexicographically (string order),
        # not numerically. For 40 trajectories named trajectory_0...trajectory_39,
        # the last 20% validation trajectories are exactly: ['trajectory_4', 'trajectory_5', 'trajectory_6', 'trajectory_7', 'trajectory_8', 'trajectory_9', 'trajectory_38', 'trajectory_39']
        all_traj_keys = sorted(list(f.keys()))
    total_traj = len(all_traj_keys)
    if total_traj < 2:
        raise ValueError("Not enough trajectories to split train/validation sets.")
    split_idx = max(1, int(0.8 * total_traj))
    train_traj_keys = all_traj_keys[:split_idx]
    val_traj_keys = all_traj_keys[split_idx:]
    if len(val_traj_keys) == 0:
        val_traj_keys = [train_traj_keys.pop()]
    if rank == 0:
        print(f"   Total trajectories: {total_traj}")
        print(f"   Train trajectories: {len(train_traj_keys)}")
        print(f"   Validation trajectories: {len(val_traj_keys)}")

    train_dataset = DGPdeDataset(
        data_path=config['data_path'],
        train_time_steps=config['train_time_steps'],
        rank=rank,
        trajectory_keys=train_traj_keys
    )
    val_dataset = DGPdeDataset(
        data_path=config['data_path'],
        train_time_steps=config['train_time_steps'],
        rank=rank,
        trajectory_keys=val_traj_keys
    )
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = create_dg_loader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        sampler=train_sampler
    )
    val_loader = create_dg_loader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        sampler=val_sampler
    )
    
    if rank == 0:
        print("\nCreating model...")
    model = DGNet(config)
    if rank == 0:
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    loss_fn = Loss(config)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config['lr_decay_step_size'], 
        gamma=config['lr_decay_gamma']
    )
    
    trainer = DGTrainer(model, optimizer, loss_fn, config, rank=rank, local_rank=local_rank, scheduler=scheduler)

    if rank == 0:
        print("\nStarting training...")
    trainer.train(train_loader, val_loader, config['num_epochs'])
    
    if rank == 0:
        print("\nTraining complete. Architecture validation succeeded.")

    cleanup_ddp()


if __name__ == "__main__":
    main() 