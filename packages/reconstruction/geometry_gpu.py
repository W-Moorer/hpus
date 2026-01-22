
import torch
from typing import Tuple

def point_triangle_distance(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """
    Calculate the shortest distance from each point to the set of triangles on GPU.
    
    Args:
        points: (M, 3) tensor
        triangles: (N, 3, 3) tensor
        
    Returns:
        (M,) tensor of shortest distances.
    """
    # points: (M, 3) -> (M, 1, 3)
    P = points.unsqueeze(1) 
    
    # faces: (N, 3, 3) -> (1, N, 3, 3)
    # A, B, C: (1, N, 3)
    A = triangles[:, 0, :].unsqueeze(0)
    B = triangles[:, 1, :].unsqueeze(0)
    C = triangles[:, 2, :].unsqueeze(0)
    
    # Vectors
    E0 = B - A # (1, N, 3)
    E1 = C - A # (1, N, 3)
    E2 = P - A # (M, N, 3) - Broadcasting works here
    
    # Dot products
    # E0 * E0 -> (1, N, 3). sum(dim=2) -> (1, N)
    a00 = (E0 * E0).sum(dim=2) 
    a01 = (E0 * E1).sum(dim=2)
    a11 = (E1 * E1).sum(dim=2)
    
    # E2 * E0 -> (M, N, 3). sum(dim=2) -> (M, N)
    b0 = (E2 * E0).sum(dim=2)
    b1 = (E2 * E1).sum(dim=2)
    
    # Barycentric determinant
    det = a00 * a11 - a01 * a01
    div = torch.where(torch.abs(det) < 1e-12, torch.ones_like(det), det)
    
    # Unclamped projected coordinates (M, N)
    u = (a11 * b0 - a01 * b1) / div
    v = (a00 * b1 - a01 * b0) / div
    
    # --- Distances to Edges and Interior ---
    # 1. Edge AB (Segment param t0)
    # Closest point on line AB: P_line = A + t * E0
    # t = dot(AP, AB) / dot(AB, AB) = b0 / a00
    t0 = b0 / torch.clamp(a00, min=1e-12)
    t0 = torch.clamp(t0, 0.0, 1.0) # (M, N)
    
    # Closest point on segment AB: Q_AB = A + t0 * E0
    # A is (1, N, 3), E0 is (1, N, 3), t0 is (M, N)
    # t0.unsqueeze(2) -> (M, N, 1)
    # Result -> (M, N, 3)
    Q_AB = A + t0.unsqueeze(2) * E0
    dist_AB_sq = ((P - Q_AB)**2).sum(dim=2) # (M, N)
    
    # 2. Edge AC (Segment param t1)
    # t = dot(AP, AC) / dot(AC, AC) = b1 / a11
    t1 = b1 / torch.clamp(a11, min=1e-12)
    t1 = torch.clamp(t1, 0.0, 1.0)
    Q_AC = A + t1.unsqueeze(2) * E1
    dist_AC_sq = ((P - Q_AC)**2).sum(dim=2)
    
    # 3. Edge BC (Segment param t2)
    # Line BC: B + t * (C-B)
    E_BC = C - B # (1, N, 3)
    E_BP = P - B # (M, N, 3)
    dot_BC_BC = (E_BC * E_BC).sum(dim=2) # (1, N)
    dot_BP_BC = (E_BP * E_BC).sum(dim=2) # (M, N)
    
    t2 = dot_BP_BC / torch.clamp(dot_BC_BC, min=1e-12)
    t2 = torch.clamp(t2, 0.0, 1.0)
    Q_BC = B + t2.unsqueeze(2) * E_BC
    dist_BC_sq = ((P - Q_BC)**2).sum(dim=2)
    
    # 4. Interior (Face)
    # Check if P's projection is inside properties
    # u >= 0, v >= 0, u + v <= 1.
    mask_inside = (det > 1e-12) & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
    
    # Q_face = A + u*E0 + v*E1
    Q_face = A + u.unsqueeze(2) * E0 + v.unsqueeze(2) * E1
    dist_face_sq = ((P - Q_face)**2).sum(dim=2)
    
    # If not inside, set to infinity
    dist_face_sq = torch.where(mask_inside, dist_face_sq, torch.tensor(float('inf'), device=points.device))
    
    # Min of all
    min_sq = torch.min(dist_AB_sq, dist_AC_sq)
    min_sq = torch.min(min_sq, dist_BC_sq)
    min_sq = torch.min(min_sq, dist_face_sq)
    
    # Sqrt
    distances = torch.sqrt(torch.clamp(min_sq, min=0.0))
    
    # Min over triangles
    shortest_dists, _ = torch.min(distances, dim=1)
    
    return shortest_dists

def point_segment_distance(points: torch.Tensor, segments_a: torch.Tensor, segments_b: torch.Tensor) -> torch.Tensor:
    """
    Calculate shortest distance from points to segments on GPU.
    
    Args:
        points: (M, 3)
        segments_a: (K, 3) Start points
        segments_b: (K, 3) End points
        
    Returns:
        (M,) Tensor of min distance to any segment
    """
    # P: (M, 1, 3)
    # A: (1, K, 3)
    # B: (1, K, 3)
    P = points.unsqueeze(1)
    A = segments_a.unsqueeze(0)
    B = segments_b.unsqueeze(0)
    
    # Segment vectors
    AB = B - A # (1, K, 3)
    AP = P - A # (M, N, 3)
    
    # t = dot(AP, AB) / dot(AB, AB)
    dot_AP_AB = (AP * AB).sum(dim=2) # (M, K)
    dot_AB_AB = (AB * AB).sum(dim=2) # (1, K)
    
    t = dot_AP_AB / torch.clamp(dot_AB_AB, min=1e-12)
    t = torch.clamp(t, 0.0, 1.0) # (M, K)
    
    # Closest point C = A + t*AB
    # t is (M, K), AB is (1, K, 3)
    # t.unsqueeze(2) -> (M, K, 1)
    # AB -> (1, K, 3) broadcast ok
    C = A + t.unsqueeze(2) * AB # (M, K, 3)
    
    # Dist sq
    dist_sq = ((P - C)**2).sum(dim=2) # (M, K)
    
    # Min over segments
    min_dist_sq, _ = torch.min(dist_sq, dim=1) # (M,)
    
    return torch.sqrt(torch.clamp(min_dist_sq, min=0.0))
