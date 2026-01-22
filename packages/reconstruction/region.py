
import numpy as np
import trimesh
import torch
from typing import Tuple, Optional, List
from scipy.spatial import cKDTree
from .solver import NormalDirectionalSolverGPU
from .pou import C2Bump, compute_weights_batch
from .sampling import poisson_disk_sampling

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
                 lambda_penalty: float = 10.0,
                 gating_margin_ratio: float = 1.0):
        """
        Args:
            region_id: Unique ID of the region.
            mesh: Trimesh object for this region.
            device: 'cuda' or 'cpu'.
            patch_radius_ratio: Ratio of patch radius to average point spacing.
            lambda_penalty: Penalty strength for points outside validity domain.
            gating_margin_ratio: Ratio of spacing to extend validity domain (Plateau).
        """
        self.region_id = region_id
        self.mesh = mesh
        self.device = device
        self.lambda_penalty = lambda_penalty
        
        # Patch parameters
        self.patch_centers: np.ndarray = None
        self.patch_radii: np.ndarray = None
        self.solvers: List[NormalDirectionalSolverGPU] = []
        
        # Anchors for Jurisdiction Gating
        self.anchors: np.ndarray = None
        self.anchor_tree: cKDTree = None
        
        # Fitting parameters
        self.patch_radius_ratio = patch_radius_ratio
        self.gating_margin_ratio = gating_margin_ratio
        
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
        Generate patch centers using Poisson Disk Sampling and adaptive radii.
        """
        vertices = np.ascontiguousarray(self.mesh.vertices)
        
        # 1. Determine number of patches
        if num_patches <= 0:
            # Heuristic: 1 patch per 10 vertices (patch_divisor = 10.0)
            # Ensure at least 1 patch
            num_patches = max(1, int(len(vertices) / 10.0))
        
        # 2. Sample patch centers (Poisson Disk)
        # Note: self.mesh.vertices might be sparse or dense. 
        # If mesh is very low poly, we might need to sample on surface first to get enough candidates.
        # But here we assume vertices are dense enough or we just use vertices.
        # For better quality, let's sample points on surface first if vertices are few?
        # Actually, poisson_disk_sampling takes points. 
        # Let's pass mesh vertices directly for now.
        self.patch_centers = poisson_disk_sampling(vertices, num_patches)
        
        # 3. Adaptive Radius Strategy
        num_p = len(self.patch_centers)
        if num_p < 1:
             # Fallback if sampling failed completely (unlikely)
             self.patch_centers = np.mean(vertices, axis=0, keepdims=True)
             num_p = 1

        if num_p >= 2:
            # --- Phase 1: Initial Radius (k-NN based) ---
            tree_p = cKDTree(self.patch_centers)
            # Query 2 nearest neighbors (1st is self, 2nd is closest other)
            dists_p, _ = tree_p.query(self.patch_centers, k=2)
            # Use max of nearest neighbor distances as baseline scale
            tau_val = np.max(dists_p[:, 1])
            
            # Initial radius: slightly larger than spacing to ensure overlap
            # (1.0 + 1.0) * tau / 2.0  -> effectively tau
            initial_rho = tau_val
            self.patch_radii = np.full(num_p, initial_rho)
            
            # --- Phase 2: Density Adjustment ---
            # Ensure each patch covers at least n_min vertices
            n_min = 10
            tree_v = cKDTree(vertices)
            
            # Batch query might be memory heavy if num_p is huge, loop is safer
            for m in range(num_p):
                # Check count in current radius
                # query_ball_point is fast
                indices = tree_v.query_ball_point(self.patch_centers[m], self.patch_radii[m])
                count = len(indices)
                
                if count < n_min:
                    # Expand to include n_min-th neighbor
                    # k = min(len(vertices), n_min)
                    k_query = min(len(vertices), n_min)
                    dists_v, _ = tree_v.query(self.patch_centers[m], k=k_query)
                    
                    # New radius = dist to k-th neighbor * margin
                    # dists_v can be array if k>1
                    if isinstance(dists_v, np.ndarray):
                        new_r = dists_v[-1] * 1.05
                    else:
                        new_r = dists_v * 1.05
                        
                    self.patch_radii[m] = max(self.patch_radii[m], new_r)
            
            # --- Phase 3: Global Coverage Adjustment ---
            # Ensure every vertex is covered by at least one patch
            # Query nearest patch for each vertex
            # tree_p is already built
            v_dists, v_indices = tree_p.query(vertices, k=1)
            
            expanded_count = 0
            for i in range(len(vertices)):
                closest_p_idx = v_indices[i]
                dist_to_p = v_dists[i]
                
                if dist_to_p > self.patch_radii[closest_p_idx]:
                    # Expand this patch to cover this vertex
                    new_r = dist_to_p * 1.05
                    self.patch_radii[closest_p_idx] = max(self.patch_radii[closest_p_idx], new_r)
                    expanded_count += 1
            
            print(f"[Region {self.region_id}] Setup {num_p} patches. "
                  f"Radius: mean={np.mean(self.patch_radii):.4f}, "
                  f"expanded {expanded_count} times for coverage.")
                  
        else:
            # Single patch case: Cover entire bounding box
            bbox_min = np.min(vertices, axis=0)
            bbox_max = np.max(vertices, axis=0)
            bbox_diag = np.linalg.norm(bbox_max - bbox_min)
            # If vertices are essentially a point
            if bbox_diag < 1e-6: bbox_diag = 1.0
            
            self.patch_radii = np.array([bbox_diag * 0.8]) # 0.8 covers center to corner approx
            print(f"[Region {self.region_id}] Single patch setup.")

        # Store anchors for gating
        self.anchors = vertices
        self.anchor_tree = cKDTree(self.anchors)

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
            self.patch_radii, # Pass full array of radii
            device=self.device
        )
        return vals

    def compute_gating(self, points: np.ndarray) -> np.ndarray:
        """
        Compute chi_i(x).
        Uses a Plateau-based function:
        - 1.0 if dist < margin
        - Decays to 0.0 otherwise.
        """
        _, dists, _ = self.mesh.nearest.on_surface(points)
        
        # Gating bandwidth
        sigma = self.avg_spacing * 5.0 # Falloff width
        margin = self.avg_spacing * self.gating_margin_ratio # Trust region
        
        # Plateau logic
        # If dist < margin, effective_dist = 0 -> chi = 1
        effective_dist = np.maximum(0, dists - margin)
        
        chi = np.exp( - (effective_dist / sigma)**2 )
        return chi

    def evaluate_gated(self, points: np.ndarray) -> np.ndarray:
        """
        Returns tilde_F_i(x) = F_i(x) + lambda * (1 - chi_i(x))
        """
        f_i = self.evaluate(points)
        chi_i = self.compute_gating(points)
        
        # If lambda is large, this drives value huge outside region.
        return f_i + self.lambda_penalty * (1.0 - chi_i)

    def compute_dist_to_region(self, points: np.ndarray) -> np.ndarray:
        """
        Compute minimum distance to region anchors.
        """
        dists, _ = self.anchor_tree.query(points, k=1)
        return dists


