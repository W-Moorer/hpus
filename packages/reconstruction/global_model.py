
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
                 device: str = 'cpu'):
        
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

    def evaluate(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate global field F(x).
        
        Returns:
            F: (N,) global field values
            grad_F: (N, 3) gradients (finite diff or analytical if implemented)
            normals: (N, 3) normalized gradients
        """
        # 1. Identify Primary Region i0(x) for each point
        # topology.get_closest_region returns: dists, pts, region_ids
        _, _, primary_rids = self.topology.get_closest_region(points)
        
        # 2. Computation is point-wise (or batched by primary region)
        # To vectorise, we group points by their primary region.
        
        N = len(points)
        F_global = np.zeros(N)
        
        unique_regions = np.unique(primary_rids)
        
        for r_id in unique_regions:
            mask = (primary_rids == r_id)
            indices = np.where(mask)[0]
            pts_sub = points[mask]
            
            # --- Per-group evaluation ---
            
            # Primary Region Field
            region_obj = self.region_map[r_id]
            # E_i0 = tilde_F_i0 (penalty extended)
            # Use gated evaluation for primary region too? 
            # PDF says: E_i0 = tilde_F_i0.
            val_i0 = region_obj.evaluate_gated(pts_sub)
            
            # Collect Neighbors
            # Adj(i0)
            neighbors = list(self.topology.region_adjacency.neighbors(r_id))
            
            if not neighbors:
                # No neighbors, just single region
                F_global[mask] = val_i0
                continue
                
            # Candidate Energies
            # candidates = [i0] + neighbors
            
            # We need beta_{i0->j} for each neighbor
            vals_candidates = [val_i0]
            cand_indices = [r_id] # To track which region gave which val
            
            betas_max = np.zeros(len(pts_sub))
            
            # For each neighbor
            for n_id in neighbors:
                # Distance to boundary Gamma_{i0->j}
                distar = self.topology.get_boundary_distance(r_id, n_id, pts_sub)
                
                # Beta computation
                # beta = rho(delta / h)
                t = distar / self.h
                # C2 Bump: (1-t)^4(4t+1) if t<1 else 0
                beta = np.zeros_like(t)
                in_range = t < 1.0
                if np.any(in_range):
                    ti = t[in_range]
                    beta[in_range] = (1 - ti)**4 * (4 * ti + 1)
                
                # Update Max Beta (for epsilon)
                betas_max = np.maximum(betas_max, beta)
                
                # E_j = tilde_F_j + Lambda(1 - beta)
                n_obj = self.region_map[n_id]
                val_j_raw = n_obj.evaluate_gated(pts_sub)
                val_j_gated = val_j_raw + self.Lambda * (1.0 - beta)
                
                vals_candidates.append(val_j_gated)
                cand_indices.append(n_id)
            
            # 3. Jurisdiction Gating (Relative Distance-based)
            # Find distances to all candidate regions
            all_cands = [r_id] + neighbors
            M = len(pts_sub)
            K_c = len(all_cands)
            cand_dists = np.zeros((M, K_c))
            
            for i, c_id in enumerate(all_cands):
                cand_dists[:, i] = self.region_map[c_id].compute_dist_to_region(pts_sub)
            
            # Global min dist at each point among candidates
            global_min_dist = np.min(cand_dists, axis=1)
            
            # Jurisdicton bandwidth (buffer zone)
            # delta_jur = max(2.0 * self.h, 0.15) 
            # Use h as reference
            delta_jur = self.h * 2.0
            
            # 4. Soft-Min with Jurisdiction Gating
            # epsilon(x) = eps_far + (eps_edge - eps_far) * beta_max
            eps_x = self.eps_far + (self.eps_edge - self.eps_far) * betas_max
            
            # Final energies incorporating both topology gating (beta) and jurisdiction gating
            E_final_list = []
            
            # Primary region (index 0 in all_cands)
            dist_diff_0 = np.clip(cand_dists[:, 0] - global_min_dist, 0, None)
            u_jur_0 = np.clip(dist_diff_0 / delta_jur, 0, 1)
            w_jur_0 = 1.0 - (6 * u_jur_0**5 - 15 * u_jur_0**4 + 10 * u_jur_0**3) # C2 step
            
            # Penalty = Lambda * (1 - w_jur)
            # This ensures even the primary region is suppressed if it's far from its own anchors
            # (though primary usually is the closest)
            E_final_list.append(val_i0 + self.Lambda * (1.0 - w_jur_0))
            
            # Neighbors
            for i, n_id in enumerate(neighbors):
                # idx in cand_dists is i+1
                dist_diff_n = np.clip(cand_dists[:, i+1] - global_min_dist, 0, None)
                u_jur_n = np.clip(dist_diff_n / delta_jur, 0, 1)
                w_jur_n = 1.0 - (6 * u_jur_n**5 - 15 * u_jur_n**4 + 10 * u_jur_n**3)
                
                # neighbor energy Ej = val_j_gated + Lambda * (1 - w_jur)
                # val_j_gated already has hierarchy penalty Lambda(1-beta)
                Ej_final = vals_candidates[i+1] + self.Lambda * (1.0 - w_jur_n)
                E_final_list.append(Ej_final)
            
            # Stack final candidates: (K, M)
            E_stack = np.stack(E_final_list, axis=0)
            
            # Log-Sum-Exp for Soft-Min
            E_min = np.min(E_stack, axis=0)
            E_shifted = E_stack - E_min[np.newaxis, :]
            eps_expanded = eps_x[np.newaxis, :]
            
            arg = -E_shifted / np.maximum(eps_expanded, 1e-12)
            exp_term = np.exp(arg)
            sum_exp = np.sum(exp_term, axis=0)
            
            F_group = -eps_x * np.log(np.maximum(sum_exp, 1e-12)) + E_min
            F_global[mask] = F_group
            
        return F_global

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
