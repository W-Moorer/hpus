
import numpy as np
import trimesh
import torch
from typing import Tuple, Optional, List
from .solver import NormalDirectionalSolverGPU
from .pou import C2Bump, compute_weights_batch

class RegionReconstructor:
    """
    Reconstructs the implicit surface for a single smooth region using HRBF-PoU.
    
    Features:
    - Patch generation (Poisson disk or simple sampling).
    - Local HRBF solving (Value + Normal constraints).
    - Region Gating (Penalty for points far from region).
    """

    def __init__(self, region_id: int, mesh: trimesh.Trimesh, 
                 device: str = 'cuda',
                 patch_radius_ratio: float = 3.0,
                 lambda_penalty: float = 10.0):
        """
        Args:
            region_id: Unique ID of the region.
            mesh: Trimesh object for this region.
            device: 'cuda' or 'cpu'.
            patch_radius_ratio: Ratio of patch radius to average point spacing.
            lambda_penalty: Penalty strength for points outside validity domain.
        """
        self.region_id = region_id
        self.mesh = mesh
        self.device = device
        self.lambda_penalty = lambda_penalty
        
        # Patch parameters
        self.patch_centers: np.ndarray = None
        self.patch_radii: np.ndarray = None
        self.solvers: List[NormalDirectionalSolverGPU] = []
        
        # Fitting parameters
        self.patch_radius_ratio = patch_radius_ratio
        
        # Estimate average spacing for radius
        # Approximate by sqrt(area / N)
        if len(self.mesh.vertices) > 0:
            area = self.mesh.area
            n_verts = len(self.mesh.vertices)
            self.avg_spacing = np.sqrt(area / (n_verts if n_verts > 0 else 1))
        else:
            self.avg_spacing = 0.1 # Fallback

    def setup_patches(self, num_patches: int = -1):
        """
        Generate patch centers and radii.
        If num_patches is -1, heuristic is used.
        """
        if num_patches <= 0:
            # Heuristic: 1 patch per 5-10 vertices? Or Poisson Disk?
            # Let's verify sample_surface.
            # For accurate reconstruction, we want patches to overlap.
            num_patches = max(10, len(self.mesh.vertices) // 5)
        
        # Sample points on surface for patch centers
        self.patch_centers, _ = trimesh.sample.sample_surface(self.mesh, num_patches)
        
        # Determine radius: k-NN distance or fixed ratio
        # Simple fixed ratio based on density
        # Volume / patches approx?
        # Let's use avg spacing.
        radius = self.avg_spacing * self.patch_radius_ratio
        self.patch_radii = np.full(len(self.patch_centers), radius)
        
        print(f"[Region {self.region_id}] Setup {len(self.patch_centers)} patches, Radius={radius:.4f}")

    def fit(self, global_sdf_fn=None):
        """
        Solve local HRBF for each patch.
        
        Args:
            global_sdf_fn: Optional function(points) -> (sdf, grad). 
                           If None, uses mesh normals and 0-value on surface.
        """
        self.solvers = []
        
        # Neighbor query for collecting constraints
        # We need constraints within support radius of each patch.
        # Use a KDTree on mesh vertices for constraint collection.
        # (Better: Sample points on mesh more densely, then query)
        
        # Constraints: mesh vertices are good constraints.
        constraints_p = np.ascontiguousarray(self.mesh.vertices)
        constraints_n = np.ascontiguousarray(self.mesh.vertex_normals)
        
        # Build KDTree for constraints
        from scipy.spatial import cKDTree
        tree = cKDTree(constraints_p)
        
        for i, center in enumerate(self.patch_centers):
            radius = self.patch_radii[i]
            
            # Find constraints in this patch
            # Query constraints within support radius
            idx = tree.query_ball_point(center, radius * 1.5) # 1.5x overlap for constraints
            
            if len(idx) < 10:
                # Not enough constraints, maybe expand?
                # Enough constraints?
                k_query = min(15, len(constraints_p))
                d, idx_k = tree.query(center, k=k_query)
                idx = idx_k # Use k-nn indices
            
            # Ensure indices are numpy int array
            idx = np.asarray(idx, dtype=np.int64)
            
            p_local = constraints_p[idx]
            n_local = constraints_n[idx]
            
            # Create solver
            solver = NormalDirectionalSolverGPU(device=self.device)
            solver.solve(p_local, n_local) # f=0, g=1 by default
            
            self.solvers.append(solver)
            
        print(f"[Region {self.region_id}] Fitted {len(self.solvers)} local solvers.")

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate F_i(x) using PoU blending.
        """
        # Batch evaluate all solvers?
        # That logic is already effectively in `batch_evaluate_all_solvers` in solver_gpu.py
        # But we need to call it.
        
        from .solver import batch_evaluate_all_solvers
        
        # We need to pass the list of solvers.
        # But `batch_evaluate_all_solvers` assumes list of objects.
        
        vals = batch_evaluate_all_solvers(
            points, 
            self.solvers, 
            self.patch_centers, 
            self.patch_radii[0], # Assuming uniform radius for now as per solver signature
            device=self.device
        )
        return vals

    def compute_gating(self, points: np.ndarray) -> np.ndarray:
        """
        Compute chi_i(x).
        Simple implementation: 1 if close to mesh, decays to 0.
        Uses nearest point distance to region mesh.
        """
        # Query distance to mesh
        # For batch, trimesh.nearest.on_surface is okay but maybe slow for massive queries.
        # Use our pre-computed avg_spacing as a scale.
        
        # We assume points are close-ish if we are calling this? 
        # Actually this is called globally.
        pass # Trimesh nearest is fast enough for prototyping.
        
        # Optim: use pre-calculated SDF? or chamfer?
        # Let's use Mesh BVH.
        
        _, dists, _ = self.mesh.nearest.on_surface(points)
        
        # Gating bandwidth
        sigma = self.avg_spacing * 5.0 # Validity domain
        
        # chi = exp(- (d/sigma)^2 )? 
        # Or strict cutoff? PDF says "1 in core, 0 outside".
        # Smooth transition is better.
        
        # Let's use a C2 bump or sigmoid.
        # chi = 1 if d < sigma, else decay.
        
        # Let's implement simple penalty based on distance.
        # If dist > sigma, chi -> 0.
        
        # chi(d) = 1.0 / (1.0 + (d/sigma)**4) # Pseudo-Butterworth
        chi = np.exp( - (dists / sigma)**2 )
        return chi

    def evaluate_gated(self, points: np.ndarray) -> np.ndarray:
        """
        Returns tilde_F_i(x) = F_i(x) + lambda * (1 - chi_i(x))
        """
        f_i = self.evaluate(points)
        chi_i = self.compute_gating(points)
        
        # If lambda is large, this drives value huge outside region.
        return f_i + self.lambda_penalty * (1.0 - chi_i)

