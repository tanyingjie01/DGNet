"""DGNet neural modules for operator correction and residual dynamics."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
import math

from torch_geometric.nn import MLP, MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree


class MPNNLayer(MessagePassing):
    """Single message-passing layer."""
    
    def __init__(self, 
                 node_dim: int,
                 edge_dim: int,
                 hidden_dim: int,
                 aggregation: str = 'mean'):
        """Initialize one MPNN layer."""
        super().__init__(aggr=aggregation)

        self.node_feature_dim = node_dim
        self.edge_feature_dim = edge_dim
        self.hidden_dim = hidden_dim

        self.message_mlp = MLP(
            in_channels=2 * node_dim + edge_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=2,
            act='relu'
        )
        
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
        """Run one message-passing update."""
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.shape[1], self.edge_feature_dim, device=edge_index.device)

        return self.propagate(edge_index, x=node_features, edge_attr=edge_attr)
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """Build edge message."""
        message_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(message_input)
    
    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Update node state."""
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(update_input)


class MPNNProcessor(nn.Module):
    """Stacked MPNN processor."""
    
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 hidden_dim: int,
                 num_layers: int = 3,
                 aggregation: str = 'mean',
                 residual: bool = True):
        """Initialize stacked message-passing layers."""
        super().__init__()
        
        self.num_layers = num_layers
        self.residual = residual
        
        self.layers = nn.ModuleList([
            MPNNLayer(node_dim, edge_dim, hidden_dim, aggregation)
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(node_dim) for _ in range(num_layers)
        ])
    
    def forward(self,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run stacked message passing."""
        h = node_features

        for layer, norm in zip(self.layers, self.layer_norms):
            h_new = layer(h, edge_index, edge_attr)

            if self.residual:
                h = h + h_new
            else:
                h = h_new

            h = norm(h)
        
        return h

class OperatorCorrector(nn.Module):
    """Learn operator corrections on graph edges."""
    
    def __init__(self,
                 spatial_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 operator_dim: int = 1,
                 num_node_types: int = 3):
        """Initialize operator corrector."""
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.operator_dim = operator_dim
        self.num_node_types = num_node_types
        
        self.node_encoder = MLP(
            in_channels=spatial_dim + 1 + num_node_types,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=2,
            act='relu'
        )
        
        self.edge_encoder = MLP(
            in_channels=spatial_dim + 1,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=2,
            act='relu'
        )
        
        self.processor = MPNNProcessor(
            node_dim=hidden_dim,
            edge_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        self.edge_corrector = MLP(
            in_channels=2 * hidden_dim, 
            hidden_channels=hidden_dim,
            out_channels=operator_dim,
            num_layers=2,
            act='relu'
        )
    
    def forward(self, 
                graph_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predict a correction matrix on graph edges."""

        nodes = graph_data['nodes']
        unique_edges = graph_data['edges']
        node_volumes = graph_data.get('node_volumes', torch.ones(nodes.shape[0], device=nodes.device))
        node_type = graph_data.get('node_type', torch.zeros(nodes.shape[0], device=nodes.device))

        N = nodes.shape[0]
        device = nodes.device

        edge_reversed = unique_edges.flip(dims=[1])
        edges_bidirectional = torch.cat([unique_edges, edge_reversed], dim=0)
        edge_index = edges_bidirectional.T

        src_coords = nodes[edges_bidirectional[:, 0]]
        dst_coords = nodes[edges_bidirectional[:, 1]]
        coord_diffs = dst_coords - src_coords
        distances = torch.norm(coord_diffs, dim=1, keepdim=True)

        edge_features = torch.cat([coord_diffs, distances], dim=1)
        encoded_edges = self.edge_encoder(edge_features)

        node_type_onehot = F.one_hot(node_type.long(), num_classes=self.num_node_types).float()
        node_input = torch.cat([nodes, node_volumes.unsqueeze(-1), node_type_onehot], dim=-1)
        encoded_nodes = self.node_encoder(node_input)

        processed_nodes = self.processor(encoded_nodes, edge_index, encoded_edges)
        src_nodes = processed_nodes[edges_bidirectional[:, 0]]
        dst_nodes = processed_nodes[edges_bidirectional[:, 1]]

        edge_inputs = torch.cat([
            src_nodes,
            dst_nodes,
        ], dim=1)

        raw_corrections = self.edge_corrector(edge_inputs)

        scaling_factor = 1e-5
        corrections = scaling_factor * torch.tanh(raw_corrections)

        delta_L = torch.zeros(N, N, device=device)

        src, dst = edges_bidirectional[:, 0], edges_bidirectional[:, 1]
        delta_L[dst, src] = corrections.squeeze(-1)

        row_sums = delta_L.sum(dim=1)
        delta_L[torch.arange(N, device=device), torch.arange(N, device=device)] = -row_sums
        
        return delta_L

class AttentionCombiner(nn.Module):
    """Temporal-spatial attention combiner."""
    
    def __init__(self,
                 spatial_dim: int,
                 feature_dim: int,
                 hidden_dim: int = 64,
                 num_heads: int = 4):
        """Initialize attention module."""
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.query_encoder = MLP(
            in_channels=feature_dim + spatial_dim + 1,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=2,
            act='relu'
        )
        
        self.key_encoder = MLP(
            in_channels=feature_dim + spatial_dim + 1,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=2,
            act='relu'
        )
        
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
        """Compute attention weights over history."""
        N = current_state.shape[0]
        T = history_states.shape[0]
        device = current_state.device

        current_time_tensor = torch.full((N, 1), current_time, device=device)
        query_input = torch.cat([current_state, current_coords, current_time_tensor], dim=1)
        queries = self.query_encoder(query_input)

        history_times_expanded = history_times.view(T, 1, 1).expand(T, N, 1)
        history_coords_expanded = history_coords.unsqueeze(0).expand(T, N, -1)
        key_inputs = torch.cat([
            history_states,
            history_coords_expanded,
            history_times_expanded
        ], dim=2)

        key_inputs = key_inputs.reshape(T * N, -1)
        keys = self.key_encoder(key_inputs)
        keys = keys.reshape(T, N, self.hidden_dim)
        values = keys

        queries = queries.unsqueeze(1)
        keys = keys.permute(1, 0, 2)
        values = values.permute(1, 0, 2)

        _, attention_weights = self.attention(
            query=queries,
            key=keys,
            value=values,
            need_weights=True,
            average_attn_weights=True
        )

        return attention_weights.squeeze(1)


class NonlinearDynamicsSolver(nn.Module):
    """Predict nonlinear dynamics term r(u^k)."""
    
    def __init__(self,
                 spatial_dim: int,
                 node_feature_dim: int,
                 output_dim: int,
                 hidden_dim: int = 128,
                 num_processing_layers: int = 5,
                 edge_dim: int = 8,
                 num_node_types: int = 3):
        """Initialize nonlinear solver."""
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.output_dim = output_dim
        self.num_node_types = num_node_types
        
        self.node_encoder = MLP(
            in_channels=node_feature_dim + spatial_dim + num_node_types,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=2,
            act='relu'
        )
        
        self.edge_encoder = MLP(
            in_channels=spatial_dim + node_feature_dim + 1,
            hidden_channels=hidden_dim,
            out_channels=edge_dim,
            num_layers=1,
            act='relu'
        )
        
        self.processor = MPNNProcessor(
            node_dim=hidden_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_processing_layers,
            residual=True
        )
        
        self.decoder = MLP(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            num_layers=3,
            act='relu'
        )

    def forward(self, graph_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward nonlinear dynamics solver."""
        nodes = graph_data['nodes']
        unique_edges = graph_data['edges']
        node_features = graph_data['node_features']
        node_type = graph_data.get('node_type', torch.zeros(nodes.shape[0], device=nodes.device))
        
        edge_reversed = unique_edges.flip(dims=[1])
        edges_bidirectional = torch.cat([unique_edges, edge_reversed], dim=0)
        edge_index = edges_bidirectional.T
        
        node_type_onehot = F.one_hot(node_type.long(), num_classes=self.num_node_types).float()
        node_input = torch.cat([node_features, nodes, node_type_onehot], dim=1)
        encoded_nodes = self.node_encoder(node_input)
        
        src_coords = nodes[edges_bidirectional[:, 0]]
        dst_coords = nodes[edges_bidirectional[:, 1]]
        src_features = node_features[edges_bidirectional[:, 0]]
        dst_features = node_features[edges_bidirectional[:, 1]]
        coord_diffs = dst_coords - src_coords
        feature_diffs = dst_features - src_features
        distances = torch.norm(coord_diffs, dim=1, keepdim=True)
        edge_features = torch.cat([coord_diffs, feature_diffs, distances], dim=1)
        encoded_edges = self.edge_encoder(edge_features)
        
        processed_nodes = self.processor(encoded_nodes, edge_index, encoded_edges)

        r_uk = self.decoder(processed_nodes)
        
        return r_uk

class ResidualSolver(nn.Module):
    """Residual correction solver for data path."""
    
    def __init__(self,
                 spatial_dim: int,
                 node_feature_dim: int,
                 output_dim: int,
                 hidden_dim: int = 128,
                 num_processing_layers: int = 5,
                 edge_dim: int = 8,
                 num_node_types: int = 3):
        """Initialize residual solver."""
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.output_dim = output_dim
        self.num_node_types = num_node_types
        
        self.node_encoder = MLP(
            in_channels=node_feature_dim + spatial_dim + num_node_types,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=2,
            act='relu'
        )
        
        self.edge_encoder = MLP(
            in_channels=spatial_dim + node_feature_dim + 1,
            hidden_channels=hidden_dim,
            out_channels=edge_dim,
            num_layers=1,
            act='relu'
        )
        
        self.processor = MPNNProcessor(
            node_dim=hidden_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_processing_layers,
            residual=True
        )
        
        self.decoder = MLP(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            num_layers=3,
            act='relu'
        )
    
    def forward(self, graph_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward residual solver."""

        nodes = graph_data['nodes']
        unique_edges = graph_data['edges']
        node_features = graph_data['node_features']
        boundary_info = graph_data.get('boundary_info', {})
        node_type = graph_data.get('node_type', torch.zeros(nodes.shape[0], device=nodes.device))

        N = nodes.shape[0]
        device = nodes.device

        edge_reversed = unique_edges.flip(dims=[1])
        edges_bidirectional = torch.cat([unique_edges, edge_reversed], dim=0)
        edge_index = edges_bidirectional.T

        node_type_onehot = F.one_hot(node_type.long(), num_classes=self.num_node_types).float()
        node_input = torch.cat([node_features, nodes, node_type_onehot], dim=1)

        encoded_nodes = self.node_encoder(node_input)
        h_anchor = encoded_nodes.clone()

        src_coords = nodes[edges_bidirectional[:, 0]]
        dst_coords = nodes[edges_bidirectional[:, 1]]
        src_features = node_features[edges_bidirectional[:, 0]]
        dst_features = node_features[edges_bidirectional[:, 1]]
        coord_diffs = dst_coords - src_coords
        feature_diffs = dst_features - src_features
        distances = torch.norm(coord_diffs, dim=1, keepdim=True)
        edge_features = torch.cat([coord_diffs, feature_diffs, distances], dim=1)

        encoded_edges = self.edge_encoder(edge_features)

        h_current = encoded_nodes
        dirichlet_indices = boundary_info.get('dirichlet', {}).get('indices')

        for i in range(self.processor.num_layers):
            h_updated = self.processor.layers[i](h_current, edge_index, encoded_edges)

            if self.processor.residual:
                h_current = h_current + h_updated
            else:
                h_current = h_updated
            h_current = self.processor.layer_norms[i](h_current)

            if dirichlet_indices is not None:
                h_current[dirichlet_indices] = h_anchor[dirichlet_indices]
        
        processed_nodes = h_current
        
        u_net = self.decoder(processed_nodes)
        
        return u_net
