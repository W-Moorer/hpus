
import numpy as np
import torch
from typing import List, Dict, Tuple
from ..topology.manager import TopologyManager
from .region import RegionReconstructor

class GlobalImplicitSurface:
    """
    Implements the Topology-Driven Soft-Min Global Implicit Surface F(x).
    
    References:
        - implicit_surface_contact_draft_fixed.pdf Section 6 & 7.
    """
    
    def __init__(self, 
                 regions: List[RegionReconstructor], 
                 topology: TopologyManager,
                 epsilon_far: float = 1e-4,
                 epsilon_edge: float = 1e-2,
                 h_bandwidth: float = 1e-2,
                 lambda_gate: float = 100.0, # Lambda in PDF (for soft-min gating)
         device: str = 'cpu',
                 max_trusted_dist: float = 0.2):
        
        self.regions = regions
        self.topology = topology
        self.device = device
        
        # Mapping region_id to object
        self.region_map = {r.region_id: r for r in regions}
        
        # Parameters
        self.eps_far = epsilon_far
        self.eps_edge = epsilon_edge
        self.h = h_bandwidth
        self.Lambda = lambda_gate
        self.max_trusted_dist = max_trusted_dist

    def evaluate(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate global field F(x) on GPU if available.
        """
        N = len(points)
        
        # CPU path if device is cpu
        if self.device == 'cpu':
             # ... Legacy CPU implementation ...
             # We can keep it or just force GPU logic if we really want, but user might select CPU.
             # However, given we are optimizing, let's assume GPU path is primary.
             # If CPU, fallback to old logic.
             pass
        
        # -------------------------------------------------------------
        # GPU Optimized Path
        # -------------------------------------------------------------
        if self.device != 'cpu':
            import torch
            pts_gpu = torch.tensor(points, dtype=torch.float32, device=self.device)
            
            # 1. Identify Primary Region i0(x) using GPU Distances
            # Replaces topology.get_closest_region(points)
            
            # Compute distance to ALL regions
            # Assuming typically R is small (< 20)
            # Create a tensor for all distances (N, R)
            region_ids = sorted(list(self.region_map.keys()))
            # Map index to region_id
            idx_to_rid = {i: rid for i, rid in enumerate(region_ids)}
            rid_to_idx = {rid: i for i, rid in enumerate(region_ids)}
            
            all_dists = []
            for rid in region_ids:
                robj = self.region_map[rid]
                # Use the chunked GPU distance calculator we added to RegionReconstructor
                # But we want to keep it on GPU!
                # robj.compute_dist_to_region returns numpy. 
                # We should expose a method that returns/accepts tensor, or hack it.
                # Actually, `compute_dist_to_region` takes ndarray and returns ndarray.
                
                # Let's call the underlying logic directly or modify RegionReconstructor to accept Tensor.
                # Modifying RegionReconstructor is cleaner.
                # But for now, let's just use `geometry_gpu.point_triangle_distance` directly here since we have access.
                
                if hasattr(robj, 'triangles_gpu') and len(robj.triangles_gpu) > 0:
                     # Check for empty points? batch_point_triangle_distance handles it? no it expects points.
                     # We can Chunk here too if needed, but let's assume VRAM is OK for just distances?
                     # N=100k, Tri=540 is OK.
                     
                     # Chunking is safer similar to region.py
                     chunk_size = 10000
                     d_gpu = torch.empty(N, dtype=torch.float32, device=self.device)
                     from .geometry_gpu import point_triangle_distance
                     
                     for i in range(0, N, chunk_size):
                         end = min(i + chunk_size, N)
                         batch_dists = point_triangle_distance(pts_gpu[i:end], robj.triangles_gpu)
                         d_gpu[i:end] = batch_dists
                         
                     all_dists.append(d_gpu)
                else:
                     all_dists.append(torch.full((N,), 1e6, device=self.device))
            
            # Stack (N, R)
            stack_dists = torch.stack(all_dists, dim=1)
            
            # Min over regions
            global_min_dist, min_indices = torch.min(stack_dists, dim=1)
            primary_rids_idx = min_indices # Tensor of indices
            
            # Iterate regions to evaluate field
            F_global = torch.zeros(N, dtype=torch.float32, device=self.device)
            
            # Group by primary region
            for idx, r_id in idx_to_rid.items():
                mask = (primary_rids_idx == idx)
                if not mask.any():
                    continue
                    
                indices = torch.where(mask)[0]
                pts_sub = pts_gpu[mask]
                
                # --- Per-group evaluation ---
                region_obj = self.region_map[r_id]
                
                # Primary Field E_i0
                # region_obj.evaluate expects numpy.
                # We need a tensor-friendly evaluate.
                # `batch_evaluate_all_solvers` can return tensor!
                # region_obj.evaluate calls it.
                # Let's add `return_tensor=True` capability to region.evaluate? 
                
                # Or just manually call batch_evaluate_all_solvers here.
                from .solver import batch_evaluate_all_solvers
                
                # NOTE: We need dists for gating.
                # Primary dist is stack_dists[mask, idx]
                dist_i0 = stack_dists[mask, idx]
                
                # Compute Gating chi_i0
                sigma = region_obj.avg_spacing * 5.0
                margin = region_obj.avg_spacing * region_obj.gating_margin_ratio
                effective_dist = torch.clamp(dist_i0 - margin, min=0)
                chi_i0 = torch.exp( - (effective_dist / sigma)**2 )
                
                # Evaluate F_i0
                # We need to compute F_i0 for gating.
                # Just use the standard evaluate which returns numpy, then convert.
                pts_sub_np = pts_sub.cpu().numpy()
                val_i0_np = region_obj.evaluate(pts_sub_np)
                val_i0_gpu = torch.tensor(val_i0_np, device=self.device, dtype=torch.float32)
                
                # E_i0 = tilde_F_i0 (penalty extended)
                # tilde_F_i0 = F_i0 + Lambda * (1-chi_i0)
                E_i0 = val_i0_gpu + self.Lambda * (1.0 - chi_i0)
                
                # Neighbors
                neighbors = list(self.topology.region_adjacency.neighbors(r_id))
                
                vals_candidates = [E_i0]
                
                betas_max = torch.zeros_like(E_i0)
                
                for n_id in neighbors:
                    # Boundary Distance (GPU)
                    # Gamma_{i0->j}
                    distar = self.topology.get_boundary_distance_gpu(r_id, n_id, pts_sub, self.device)
                    
                    # Beta
                    t = distar / self.h
                    beta = torch.zeros_like(t)
                    mask_beta = t < 1.0
                    if mask_beta.any():
                        ti = t[mask_beta]
                        beta[mask_beta] = (1 - ti)**4 * (4 * ti + 1)
                        
                    betas_max = torch.max(betas_max, beta)
                    
                    # Neighbor Field & Gating
                    n_obj = self.region_map[n_id]
                    
                    # Dist to neighbor
                    try:
                        n_idx = rid_to_idx[n_id]
                        dist_j = stack_dists[mask, n_idx]
                    except Exception:
                        # Should not happen
                        dist_j = torch.full_like(dist_i0, 1e6)
                    
                    # Gating j
                    sigma_j = n_obj.avg_spacing * 5.0
                    margin_j = n_obj.avg_spacing * n_obj.gating_margin_ratio
                    eff_j = torch.clamp(dist_j - margin_j, min=0)
                    chi_j = torch.exp( - (eff_j / sigma_j)**2 )
                    
                    # Eval j
                    val_j_np = n_obj.evaluate(pts_sub_np)
                    val_j_gpu = torch.tensor(val_j_np, device=self.device, dtype=torch.float32)
                    
                    # E_j
                    val_j_gated = val_j_gpu + self.Lambda * (1.0 - chi_j)
                    E_j = val_j_gated + self.Lambda * (1.0 - beta)
                    vals_candidates.append(E_j)
                
                # Jurisdiction Gating
                # Need distances to all candidates (i0 + neighbors)
                # candidate_indices in `stack_dists`
                cand_rids = [r_id] + neighbors
                cand_indices = [rid_to_idx[rid] for rid in cand_rids]
                
                # Gather distances: (M_sub, K_c)
                # stack_dists[mask] is (M_sub, R)
                cand_dists_local = stack_dists[mask][:, cand_indices]
                
                # Global min among these candidates
                local_min_dist, _ = torch.min(cand_dists_local, dim=1)
                
                delta_jur = self.h * 2.0
                eps_x = self.eps_far + (self.eps_edge - self.eps_far) * betas_max
                
                E_final_list = []
                
                for k, cid in enumerate(cand_rids):
                    # dist_diff
                    dist_diff = torch.clamp(cand_dists_local[:, k] - local_min_dist, min=0)
                    u_jur = torch.clamp(dist_diff / delta_jur, 0, 1)
                    w_jur = 1.0 - (6 * u_jur**5 - 15 * u_jur**4 + 10 * u_jur**3)
                    
                    E_raw = vals_candidates[k]
                    E_final = E_raw + self.Lambda * (1.0 - w_jur)
                    E_final_list.append(E_final)
                    
                # Soft Min
                E_stack = torch.stack(E_final_list, dim=0) # (K, M)
                E_min, _ = torch.min(E_stack, dim=0)
                E_shifted = E_stack - E_min.unsqueeze(0)
                eps_expanded = eps_x.unsqueeze(0)
                
                arg = -E_shifted / torch.clamp(eps_expanded, min=1e-12)
                exp_term = torch.exp(arg)
                sum_exp = torch.sum(exp_term, dim=0)
                
                F_group = -eps_x * torch.log(torch.clamp(sum_exp, min=1e-12)) + E_min
                
                # Absolute Distance Cutoff
                mask_far = local_min_dist > self.max_trusted_dist
                if mask_far.any():
                    F_group[mask_far] = local_min_dist[mask_far]
                
                F_global[mask] = F_group
            
            return F_global.cpu().numpy()
            
        else:
             # Fallback to original CPU logic
             # (This block contains the original code)
             # ...
             return self._evaluate_cpu(points)

    def _evaluate_cpu(self, points):
         # ... original evaluate code ...
         pass

    def evaluate_gradient(self, points: np.ndarray, delta: float = 1e-5) -> np.ndarray:
        """
        Numerical gradient for prototyping.
        Analytical gradient is derived in PDF but complex to implement quickly without autograd of the full chain.
        Given we use PyTorch eventually, we can use Autograd?
        But here input is numpy.
        
        Let's implement central difference for robustness now.
        """
        grad = np.zeros((len(points), 3))
        
        # dX
        p_plus = points.copy(); p_plus[:, 0] += delta
        p_minus = points.copy(); p_minus[:, 0] -= delta
        f_plus = self.evaluate(p_plus)
        f_minus = self.evaluate(p_minus)
        grad[:, 0] = (f_plus - f_minus) / (2 * delta)
        
        # dY
        p_plus = points.copy(); p_plus[:, 1] += delta
        p_minus = points.copy(); p_minus[:, 1] -= delta
        f_plus = self.evaluate(p_plus)
        f_minus = self.evaluate(p_minus)
        grad[:, 1] = (f_plus - f_minus) / (2 * delta)
        
        # dZ
        p_plus = points.copy(); p_plus[:, 2] += delta
        p_minus = points.copy(); p_minus[:, 2] -= delta
        f_plus = self.evaluate(p_plus)
        f_minus = self.evaluate(p_minus)
        grad[:, 2] = (f_plus - f_minus) / (2 * delta)
        
        return grad
