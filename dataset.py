"""DGNet dataset structures, transforms, and loaders."""

import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Callable
from abc import ABC, abstractmethod


class DGGraph:
    """Container for one PDE trajectory on a graph mesh."""
    
    def __init__(self, 
                 nodes: torch.Tensor,
                 edges: torch.Tensor,
                 faces: torch.Tensor,
                 node_features: torch.Tensor,
                 source_terms: torch.Tensor,
                 initial_condition: torch.Tensor,
                 time_points: torch.Tensor,
                 edge_attr: Optional[torch.Tensor] = None,
                 boundary_info: Optional[Dict[str, Any]] = None,
                 node_type: Optional[torch.Tensor] = None,
                 **kwargs):
        """Initialize graph trajectory data."""
        self.nodes = nodes
        self.edges = edges  
        self.faces = faces
        self.node_features = node_features
        self.source_terms = source_terms
        self.initial_condition = initial_condition
        self.time_points = time_points
        self.edge_attr = edge_attr
        
        self.boundary_info = boundary_info or {}
        self.node_type = node_type if node_type is not None else torch.zeros(nodes.shape[0], dtype=torch.long, device=nodes.device)

        if boundary_info:
            self._setup_boundary_conditions(boundary_info)

        self._compute_geometric_properties()

        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def _setup_boundary_conditions(self, boundary_info: Dict[str, Any]):
        """Set node-type flags from boundary metadata."""
        for bc_type, bc_data in boundary_info.items():
            if bc_type == 'dirichlet' and 'indices' in bc_data:
                indices = bc_data['indices']
                self.node_type[indices] = 1
            elif bc_type == 'neumann' and 'target_indices' in bc_data:
                target_indices = bc_data['target_indices']
                self.node_type[target_indices] = 2
    
    def get_boundary_data(self) -> Dict[str, Any]:
        """Return boundary-condition data."""
        return self.boundary_info
    
    def get_boundary_mask(self, bc_type: str) -> Optional[torch.Tensor]:
        """Return node mask for a boundary type."""
        if bc_type == 'interior':
            return self.node_type == 0
        elif bc_type == 'dirichlet':
            return self.node_type == 1
        elif bc_type == 'neumann':
            return self.node_type == 2
        elif bc_type == 'boundary':
            return self.node_type > 0
        else:
            return None
    
    def _compute_geometric_properties(self):
        """Compute geometry-derived tensors."""

        edge_i, edge_j = self.edges[:, 0], self.edges[:, 1]
        edge_vectors = self.nodes[edge_j] - self.nodes[edge_i]
        self.edge_distances = torch.norm(edge_vectors, dim=1)

        self.node_volumes = self._estimate_node_volumes()

    def _estimate_node_volumes(self):
        """Estimate per-node dual areas from triangle faces."""
        num_nodes = self.nodes.shape[0]
        device = self.nodes.device
        face_vertices = self.nodes[self.faces]

        v0, v1, v2 = face_vertices[:, 0, :], face_vertices[:, 1, :], face_vertices[:, 2, :]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        spatial_dim = self.nodes.shape[1]
        if spatial_dim == 2:
            triangle_areas = 0.5 * torch.abs(edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0])
        elif spatial_dim == 3:
            triangle_areas = 0.5 * torch.norm(torch.cross(edge1, edge2, dim=1), dim=1)
        else:
            raise ValueError(f"Volume calculation for spatial_dim={spatial_dim} not implemented.")

        area_per_vertex = triangle_areas / 3.0

        src = area_per_vertex.repeat_interleave(3)

        index = self.faces.flatten()

        node_volumes = torch.zeros(num_nodes, dtype=torch.float32, device=device)

        node_volumes.scatter_add_(0, index, src)
        
        return node_volumes
    
    def get_timestep_data(self, t_idx: int):
        """Return data at a single time index."""
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
        """Return history data up to the given time index."""
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

class BaseTransform(ABC):
    """Base transform interface."""
    
    @abstractmethod
    def __call__(self, data: DGGraph) -> DGGraph:
        """Apply transform to a DGGraph sample."""
        pass

class Normalize(BaseTransform):
    """Standard normalization transform."""
    
    def __init__(self, 
                 normalize_features: bool = True,
                 normalize_source: bool = True, 
                 normalize_coords: bool = False):
        """Initialize normalization options."""
        self.normalize_features = normalize_features
        self.normalize_source = normalize_source  
        self.normalize_coords = normalize_coords
        
        self.stats = {}
    
    def __call__(self, data: DGGraph) -> DGGraph:
        """Normalize selected tensors."""
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
        """Normalize a tensor using cached statistics."""
        if key not in self.stats:
            self.stats[key] = {
                'mean': tensor.mean(),
                'std': tensor.std() + 1e-8
            }
        
        return (tensor - self.stats[key]['mean']) / self.stats[key]['std']
    
    def _copy_data(self, data: DGGraph) -> DGGraph:
        """Clone DGGraph data."""
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
            **{key: getattr(data, key) for key in data.__dict__ 
               if key not in ['nodes', 'edges', 'faces', 'node_features', 'source_terms', 
                             'initial_condition', 'time_points', 'edge_attr', 
                             'boundary_info', 'node_type', 'edge_distances', 'node_volumes']}
        )

class AddNoise(BaseTransform):
    """Add Gaussian noise to selected fields."""
    
    def __init__(self, 
                 noise_level: float = 0.01,
                 add_to_features: bool = True,
                 add_to_source: bool = True,
                 add_to_initial: bool = True):
        """Initialize noise settings."""
        self.noise_level = noise_level
        self.add_to_features = add_to_features
        self.add_to_source = add_to_source
        self.add_to_initial = add_to_initial
    
    def __call__(self, data: DGGraph) -> DGGraph:
        """Apply additive Gaussian noise."""
        data_copy = self._copy_data(data)
        
        if self.add_to_features:
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
        """Clone DGGraph data."""
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
            **{key: getattr(data, key) for key in data.__dict__ 
               if key not in ['nodes', 'edges', 'faces', 'node_features', 'source_terms', 
                             'initial_condition', 'time_points', 'edge_attr', 
                             'boundary_info', 'node_type', 'edge_distances', 'node_volumes']}
        )

class Compose(BaseTransform):
    """Compose multiple transforms."""
    
    def __init__(self, transforms: List[BaseTransform]):
        """Store ordered transforms."""
        self.transforms = transforms
        
    def __call__(self, data: DGGraph) -> DGGraph:
        """Apply all transforms in order."""
        for transform in self.transforms:
            data = transform(data)
        return data

class DGPdeDataset(Dataset):
    """Load and chunk PDE trajectories from HDF5."""
    
    def __init__(self, 
                 data_path: str,
                 train_time_steps: int,
                 max_samples: Optional[int] = None,
                 rank: int = 0,
                 trajectory_keys: Optional[List[str]] = None):
        """Initialize dataset and pre-load trajectory chunks."""
        self.data_path = data_path
        self.train_time_steps = train_time_steps
        self.samples = []
        self.rank = rank
        self.trajectory_keys = trajectory_keys
        
        self._chunk_and_load_data()
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _chunk_and_load_data(self):
        """Read trajectories and split them into fixed-length chunks."""
        if not self.train_time_steps or self.train_time_steps <= 0:
            raise ValueError("参数 'train_time_steps' 必须是一个正整数")

        with h5py.File(self.data_path, 'r') as f:
            traj_keys = sorted(list(f.keys()))
            if self.trajectory_keys is not None:
                missing = [k for k in self.trajectory_keys if k not in f]
                if missing:
                    raise KeyError(f"HDF5中未找到轨迹: {missing}")
                traj_keys = [k for k in traj_keys if k in self.trajectory_keys]
            
            for traj_key in traj_keys:
                traj_group = f[traj_key]

                nodes = torch.from_numpy(traj_group['nodes'][:]).float()  # [T, N, spatial_dim]
                edges = torch.from_numpy(traj_group['edges'][:]).long()  # [E, 2]
                faces = torch.from_numpy(traj_group['faces'][:]).long()  # [F, 3]

                full_node_features = torch.from_numpy(traj_group['node_features'][:]).float()  # [T, N, feature_dim]
                full_source_terms = torch.from_numpy(traj_group['source_terms'][:]).float()  # [T, N, source_dim]
                full_time_points = torch.from_numpy(traj_group['time_points'][:]).float()  # [T]

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
                
                T = full_node_features.shape[0]
                M = self.train_time_steps
                num_chunks = T // M

                for i in range(num_chunks):
                    start_idx = i * M
                    end_idx = start_idx + M

                    chunk_node_features = full_node_features[start_idx:end_idx]
                    chunk_source_terms = full_source_terms[start_idx:end_idx]
                    chunk_time_points = full_time_points[start_idx:end_idx]

                    chunk_initial_condition = full_node_features[start_idx]

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
        

    def __len__(self) -> int:
        """Return number of chunks."""
        return len(self.samples)

    def __getitem__(self, index: int) -> DGGraph:
        """Return one chunk sample."""
        return self.samples[index]


def dg_collate_fn(batch_list: List[DGGraph]) -> Dict[str, Any]:
    """Pack a list of DGGraph samples into one batch dict."""
    batch_size = len(batch_list)

    sample = batch_list[0]

    batch_data = {
        'batch_size': batch_size,
        'num_nodes': sample.nodes.shape[0],
        'num_timesteps': sample.node_features.shape[0],
        'nodes': sample.nodes,
        'edges': sample.edges,
        'faces': sample.faces,
        'edge_attr': sample.edge_attr,
        'node_volumes': sample.node_volumes,
        'node_features': torch.stack([g.node_features for g in batch_list]),
        'source_terms': torch.stack([g.source_terms for g in batch_list]),
        'initial_conditions': torch.stack([g.initial_condition for g in batch_list]),
        'time_points': sample.time_points,
        'boundary_info': sample.boundary_info,
        'node_type': sample.node_type,
        'trajectory_ids': [getattr(g, 'trajectory_id', i) for i, g in enumerate(batch_list)]
    }
    
    return batch_data

def create_dg_loader(dataset: DGPdeDataset,
                     batch_size: int = 32,
                     shuffle: bool = True,
                     num_workers: int = 8,
                     pin_memory: bool = True,
                     sampler: Optional[torch.utils.data.Sampler] = None,
                     **kwargs) -> DataLoader:
    """Create a DataLoader for DG trajectory chunks."""

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
