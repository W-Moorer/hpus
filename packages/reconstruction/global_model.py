
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
            
            # 3. Soft-Min
            # epsilon(x) = eps_far + (eps_edge - eps_far) * beta_max
            eps_x = self.eps_far + (self.eps_edge - self.eps_far) * betas_max
            
            # Stack candidates: (K, M) where K = 1 + num_neighbors
            E_stack = np.stack(vals_candidates, axis=0) # (K, M)
            
            # F = -eps * log sum exp (-E / eps)
            # Numerical stability: shift by min
            # exp(-E/eps) = exp(-(E - E_min)/eps - E_min/eps)
            #             = exp(-(E-E_min)/eps) * exp(-E_min/eps)
            
            E_min = np.min(E_stack, axis=0)
            E_shifted = E_stack - E_min[np.newaxis, :]
            
            # -E/eps
            # careful with eps (M,) broadcasting
            eps_expanded = eps_x[np.newaxis, :]
            
            arg = -E_shifted / eps_expanded
            exp_term = np.exp(arg)
            sum_exp = np.sum(exp_term, axis=0)
            
            # F = -eps * ( log(sum_exp) - E_min/eps )
            #   = -eps * log(sum_exp) + E_min
            
            F_group = -eps_x * np.log(sum_exp) + E_min
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
