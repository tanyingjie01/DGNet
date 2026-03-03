"""DGNet physics operators and boundary-condition helpers."""

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
    """Build a discrete operator matrix from mesh geometry."""
    N = nodes.shape[0]
    device = nodes.device

    L = torch.zeros(N, N, dtype=torch.float32, device=device)
    
    if operator_type == 'laplace':
        L = _build_laplace_operator(nodes, edges, faces, node_volumes, edge_attr, **kwargs)
    elif operator_type == 'gradient':
        L = _build_gradient_operator(nodes, faces, **kwargs)
    elif operator_type == 'gradient_gauss':
        L = _build_gradient_operator_gauss(nodes, edges, faces, node_volumes, **kwargs)
    elif operator_type == 'fhn':
        L = _build_fhn_operator(nodes, edges, faces, node_volumes, edge_attr, **kwargs)
    else:
        raise ValueError(f"Unsupported operator type: {operator_type}")
    
    return L

def _build_laplace_operator(nodes: torch.Tensor,
                           edges: torch.Tensor,
                           faces: torch.Tensor,
                           node_volumes: torch.Tensor,
                           edge_attr: Optional[torch.Tensor] = None,
                           **kwargs) -> torch.Tensor:
    """Build a cotangent Laplace-Beltrami operator."""
    N = nodes.shape[0]
    device = nodes.device

    if edge_attr is not None:
        weights = edge_attr[:, 0] if edge_attr.dim() > 1 else edge_attr
    else:
        face_vertices = nodes[faces]
        v0, v1, v2 = face_vertices[:, 0, :], face_vertices[:, 1, :], face_vertices[:, 2, :]
        vec_01, vec_12, vec_20 = v1 - v0, v2 - v1, v0 - v2

        dot_p2 = torch.sum(-vec_20 * vec_12, dim=1)
        dot_p0 = torch.sum(-vec_01 * vec_20, dim=1)
        dot_p1 = torch.sum(-vec_12 * vec_01, dim=1)

        if nodes.shape[1] == 2:
            two_areas = torch.abs(vec_01[:, 0] * vec_12[:, 1] - vec_01[:, 1] * vec_12[:, 0])
        else:
            two_areas = torch.norm(torch.cross(vec_01, vec_12, dim=1), dim=1)

        two_areas = two_areas.clamp(min=1e-8)
        cot_values = torch.stack([dot_p0 / two_areas, dot_p1 / two_areas, dot_p2 / two_areas], dim=1)

        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

        cot_pk, cot_pi, cot_pj = cot_values[:, 2], cot_values[:, 0], cot_values[:, 1]

        indices = torch.cat([torch.stack([i, j]), torch.stack([j, i]),
                             torch.stack([j, k]), torch.stack([k, j]),
                             torch.stack([k, i]), torch.stack([i, k])], dim=1)

        values = torch.cat([cot_pk, cot_pk, cot_pi, cot_pi, cot_pj, cot_pj])

        L_cot_sparse = torch.sparse_coo_tensor(indices, values, (N, N), device=device).coalesce()
        L_cot_dense = L_cot_sparse.to_dense()
        weights = L_cot_dense[edges[:, 0], edges[:, 1]] * 0.5

    L = torch.zeros(N, N, device=device)

    for edge_idx, (i, j) in enumerate(edges):
        i, j = int(i), int(j)
        w_ij = weights[edge_idx]

        w_ij_normalized_i = w_ij / node_volumes[i]
        w_ij_normalized_j = w_ij / node_volumes[j]

        L[i, j] = -w_ij_normalized_i
        L[j, i] = -w_ij_normalized_j

    for i in range(N):
        L[i, i] = -torch.sum(L[i, :])

    conductivity = 50.0
    rho = 7850.0
    specific_heat = 450.0
    diffusion_coeff = conductivity / (rho * specific_heat)

    return -diffusion_coeff * L

def _build_gradient_operator(nodes: torch.Tensor,
                             faces: torch.Tensor,
                             **kwargs) -> torch.Tensor:
    """Build an area-averaged discrete gradient operator."""
    N = nodes.shape[0]
    F = faces.shape[0]
    device = nodes.device

    if nodes.shape[1] != 2:
        warnings.warn("Gradient operator is only implemented for 2D meshes. Returning a zero matrix.")
        return torch.zeros(N, N, device=device)

    Dx = torch.zeros(N, N, device=device)
    Dy = torch.zeros(N, N, device=device)
    node_total_areas = torch.zeros(N, device=device)

    face_vertices = nodes[faces]
    v0, v1, v2 = face_vertices[:, 0, :], face_vertices[:, 1, :], face_vertices[:, 2, :]

    two_areas = ( (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - 
                  (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0]) ).abs()

    two_areas = two_areas.clamp(min=1e-12)
    areas = two_areas / 2.0

    c_x0 = (v1[:, 1] - v2[:, 1]) / two_areas
    c_x1 = (v2[:, 1] - v0[:, 1]) / two_areas
    c_x2 = (v0[:, 1] - v1[:, 1]) / two_areas
    c_y0 = (v2[:, 0] - v1[:, 0]) / two_areas
    c_y1 = (v0[:, 0] - v2[:, 0]) / two_areas
    c_y2 = (v1[:, 0] - v0[:, 0]) / two_areas

    node_total_areas.scatter_add_(0, faces.flatten(), areas.repeat_interleave(3))

    for f_idx in range(F):
        i, j, k = faces[f_idx]

        c_face_x = torch.tensor([c_x0[f_idx], c_x1[f_idx], c_x2[f_idx]], device=device)
        c_face_y = torch.tensor([c_y0[f_idx], c_y1[f_idx], c_y2[f_idx]], device=device)

        area_weight = areas[f_idx]

        Dx[i, [i,j,k]] += area_weight * c_face_x
        Dy[i, [i,j,k]] += area_weight * c_face_y

        Dx[j, [i,j,k]] += area_weight * c_face_x
        Dy[j, [i,j,k]] += area_weight * c_face_y

        Dx[k, [i,j,k]] += area_weight * c_face_x
        Dy[k, [i,j,k]] += area_weight * c_face_y

    Dx = Dx / node_total_areas.unsqueeze(1).clamp(min=1e-12)
    Dy = Dy / node_total_areas.unsqueeze(1).clamp(min=1e-12)

    return -0.05 * Dx


def _build_gradient_operator_gauss(nodes: torch.Tensor,
                                   edges: torch.Tensor,
                                   faces: torch.Tensor,
                                   node_volumes: torch.Tensor,
                                   **kwargs) -> torch.Tensor:
    """Build a Green-Gauss discrete gradient operator."""
    N = nodes.shape[0]
    device = nodes.device

    if nodes.shape[1] != 2:
        warnings.warn("Green-Gauss gradient operator is only implemented for 2D meshes. Returning a zero matrix.")
        return torch.zeros(N, N, device=device)

    face_centroids = nodes[faces].mean(dim=1)

    edge_to_faces_map = {}
    for face_idx, face in enumerate(faces):
        v = face.tolist()
        edge1 = tuple(sorted((v[0], v[1])))
        edge2 = tuple(sorted((v[1], v[2])))
        edge3 = tuple(sorted((v[2], v[0])))
        
        edge_to_faces_map.setdefault(edge1, []).append(face_idx)
        edge_to_faces_map.setdefault(edge2, []).append(face_idx)
        edge_to_faces_map.setdefault(edge3, []).append(face_idx)

    Dx = torch.zeros(N, N, device=device)
    Dy = torch.zeros(N, N, device=device)

    for edge_tuple, face_indices in edge_to_faces_map.items():
        if len(face_indices) != 2:
            continue

        i, j = edge_tuple
        face1_idx, face2_idx = face_indices
        C1 = face_centroids[face1_idx]
        C2 = face_centroids[face2_idx]

        S_ij = torch.tensor([C2[1] - C1[1], C1[0] - C2[0]], device=device)

        Dx[i, i] += S_ij[0] * 0.5
        Dy[i, i] += S_ij[1] * 0.5
        Dx[i, j] += S_ij[0] * 0.5
        Dy[i, j] += S_ij[1] * 0.5

        Dx[j, j] -= S_ij[0] * 0.5
        Dy[j, j] -= S_ij[1] * 0.5
        Dx[j, i] -= S_ij[0] * 0.5
        Dy[j, i] -= S_ij[1] * 0.5

    safe_node_volumes = node_volumes.unsqueeze(1).clamp(min=1e-12)
    Dx = Dx / safe_node_volumes
    Dy = Dy / safe_node_volumes

    return -0.05 * Dx


def _build_fhn_operator(nodes: torch.Tensor,
                        edges: torch.Tensor,
                        faces: torch.Tensor,
                        node_volumes: torch.Tensor,
                        edge_attr: Optional[torch.Tensor] = None,
                        **kwargs) -> torch.Tensor:
    """Build the FHN operator as Laplace + gradient terms."""
    laplace_op = _build_laplace_operator(nodes, edges, faces, node_volumes, edge_attr, **kwargs)
    gradient_op = _build_gradient_operator_gauss(nodes, edges, faces, node_volumes, **kwargs)
    return laplace_op + gradient_op


def apply_bcs_to_state(u: torch.Tensor, 
                      bc_data: Dict[str, Any]) -> torch.Tensor:
    """Apply Dirichlet/Neumann boundary constraints to state tensors."""
    u_corrected = u.clone()

    if 'dirichlet' in bc_data:
        dirichlet_bc = bc_data['dirichlet']
        u_corrected = _apply_dirichlet(u_corrected, 
                                     dirichlet_bc['indices'], 
                                     dirichlet_bc['values'])

    if 'neumann' in bc_data:
        neumann_bc = bc_data['neumann']
        u_corrected = _apply_neumann(u_corrected,
                                   neumann_bc['source_indices'],
                                   neumann_bc['target_indices'])
    
    return u_corrected


def apply_bcs_to_hidden_state(h: torch.Tensor,
                             bc_data: Dict[str, Any]) -> torch.Tensor:
    """Apply boundary constraints channel-wise to hidden states."""
    if h.dim() == 2:
        h_corrected = h.clone()
        for dim in range(h.shape[1]):
            h_corrected[:, dim] = apply_bcs_to_state(h[:, dim], bc_data)
    elif h.dim() == 3:
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
    """Set boundary values at Dirichlet indices."""
    tensor_corrected = tensor.clone()

    if tensor.dim() == 1:
        tensor_corrected[indices] = values
    elif tensor.dim() == 2:
        tensor_corrected[:, indices] = values.unsqueeze(0)
    else:
        raise ValueError(f"Unsupported tensor dimension for Dirichlet BC: {tensor.dim()}")
    
    return tensor_corrected


def _apply_neumann(tensor: torch.Tensor,
                  source_indices: torch.Tensor,
                  target_indices: torch.Tensor) -> torch.Tensor:
    """Copy interior values to boundary indices for Neumann approximation."""
    tensor_corrected = tensor.clone()

    if tensor.dim() == 1:
        tensor_corrected[target_indices] = tensor_corrected[source_indices]
    elif tensor.dim() == 2:
        tensor_corrected[:, target_indices] = tensor_corrected[:, source_indices]
    else:
        raise ValueError(f"Unsupported tensor dimension for Neumann BC: {tensor.dim()}")
    
    return tensor_corrected

