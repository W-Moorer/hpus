
import trimesh
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from scipy.spatial import cKDTree

class TopologyManager:
    """
    Manages topological queries for the Partitioned HRBF-PoU reconstruction.
    
    Responsibilities:
    1. Find closest region for a query point (using Mesh BVH).
    2. Compute distance to specific boundary groups Gamma_{i->j}.
    """
    
    def __init__(self, mesh: trimesh.Trimesh, preprocessor_results: Dict):
        self.mesh = mesh
        self.regions = preprocessor_results['regions']
        self.region_adjacency = preprocessor_results['region_adjacency']
        self.face_region_map = preprocessor_results['face_region_map']
        self.sharp_edges_indices = preprocessor_results['sharp_edges']
        
        # Build Mesh BVH (Trimesh uses rtree/embree or internal)
        # We rely on mesh.nearest.on_surface which uses the internal BVH.
        self.mesh_bvh = self.mesh # Trimesh object itself acts as the query engine
        
        # Organize boundary edges by (Region, Neighbor)
        # self._boundary_groups[region_id][neighbor_id] = list of edge indices (in mesh.edges_unique)
        self._boundary_groups: Dict[int, Dict[int, List[int]]] = {}
        self._build_boundary_groups()
        
        # Pre-cache edge segment data for fast distance calculation
        # Map edge_index -> (v_start, v_end)
        self.edges_unique = self.mesh.edges_unique
        self.vertices = self.mesh.vertices
        self.edge_segments = self.vertices[self.edges_unique] # (E, 2, 3)

    def _build_boundary_groups(self):
        """
        Group boundary edges by the pair of regions they separate.
        """
        # We need to know which sharp edge connects which regions.
        # This was partly done in preprocessing, but let's reconstruct it carefully.
        
        adj_faces = self.mesh.face_adjacency
        adj_edges_idx = self.mesh.face_adjacency_edges # Indices into vertices
        
        # Create a map from vertex-pair key to unique edge index
        # edge_key_to_idx = {tuple(sorted(e)): i for i, e in enumerate(self.mesh.edges_unique)}
        # Optimized: rely on the fact that preprocessing likely preserved order? 
        # Actually trimesh face_adjacency_edges does NOT return indices into edges_unique directly.
        
        # Let's map edges to unique edges safely
        # Note: This might be slow for huge meshes, but fine for CAD.
        
        # Precompute edge centers or some hash for matching if needed.
        # Or just re-iterate sharp edges.
        
        # Let's assume preprocessing gave us indices into edges_unique?
        # In preprocessing.py: self.sharp_edges_indices ARE indices into edges_unique.
        
        # But we need to know the regions.
        # Let's iterate face adjacency again to be sure.
        
        # Optimization: use sharp_edges_indices from output if available, but we need the regions.
        # The preprocessor output didn't explicitly link edge_index -> (r1, r2).
        
        # Re-deriving locally for robustness:
        adj_angles = self.mesh.face_adjacency_angles
        is_sharp = adj_angles > np.deg2rad(1.0) # Use a very low threshold? No, use the one from preprocessing or assume coherence?
        # Better: Recalculate using the face_region_map.
        
        for i, (f1, f2) in enumerate(adj_faces):
            r1 = self.face_region_map[f1]
            r2 = self.face_region_map[f2]
            
            if r1 != r2:
                # This is a boundary edge
                edge_verts = tuple(sorted(self.mesh.face_adjacency_edges[i]))
                
                # Find unique edge index
                # (Ideally preprocessor should pass this mapping. For now, brute force search or strict assumption?
                # Trimesh face_adjacency_edges usually correspond to 'face_adjacency' which is NOT aligned with edges_unique.)
                
                # To get index in edges_unique efficiently:
                # We can construct a lookup.
                pass 
                
        # Better approach:
        # 1. Iterate all edges_unique.
        # 2. Find faces attached to this edge.
        # 3. Check their regions.
        
        # mesh.edges_face gives faces for each edge? No.
        # mesh.faces_unique_edges gives edges for each face.
        
        faces_unique_edges = self.mesh.faces_unique_edges
        
        # Iterate over all faces, check their edges
        # This is O(F).
        
        for rid in self.regions:
            self._boundary_groups[rid] = {}
        
        # We only care about sharp edges (region boundaries).
        # We can just iterate the sharp edges identified by preprocessor if we trust they separate regions.
        # But some sharp edges might be internal to a region if we didn't split on them (unlikely).
        
        # Let's assume we iterate all unique edges that are on boundaries.
        
        # Use a map: edge_idx -> (r1, r2)
        # Scan all faces.
        edge_to_regions = {}
        
        for f_idx, r_idx in enumerate(self.face_region_map):
            for e_idx in faces_unique_edges[f_idx]:
                 if e_idx not in edge_to_regions:
                     edge_to_regions[e_idx] = set()
                 edge_to_regions[e_idx].add(r_idx)
                 
        # Now filter for edges with >1 region
        count = 0
        for e_idx, r_set in edge_to_regions.items():
            if len(r_set) > 1:
                # This is a boundary edge
                r_list = list(r_set)
                # For 2-manifold, usually 2 regions.
                for i in range(len(r_list)):
                    r_self = r_list[i]
                    for j in range(len(r_list)):
                        if i == j: continue
                        r_neighbor = r_list[j]
                        
                        if r_neighbor not in self._boundary_groups[r_self]:
                            self._boundary_groups[r_self][r_neighbor] = []
                        
                        self._boundary_groups[r_self][r_neighbor].append(e_idx)
                count += 1
        
        # print(f"[TopologyManager] Indexed {count} boundary edges.")


        self.gpu_segments_cache = {} # Cache for GPU tensors (device_str -> {key -> (A, B)})

    def get_closest_region(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find the closest region for each point.
        Returns:
            distances: (N,)
            closest_points: (N, 3)
            region_ids: (N,)
        """
        # Warning: This calls trimesh CPU search. 
        # If GPU optimizations are enabled, caller should implement alternative logic
        # or we should update this to use GPU meshes if available.
        # For now, leaving as baseline.
        closest_points, distances, triangle_ids = self.mesh.nearest.on_surface(points)
        region_ids = self.face_region_map[triangle_ids]
        return distances, closest_points, region_ids

    def get_boundary_distance(self, region_id: int, neighbor_id: int, points: np.ndarray) -> np.ndarray:
        """
        Compute minimum distance from points to the boundary set Gamma_{region -> neighbor}.
        """
        if neighbor_id not in self._boundary_groups.get(region_id, {}):
            return np.full(len(points), np.inf)
            
        edge_indices = self._boundary_groups[region_id][neighbor_id]
        if not edge_indices:
            return np.full(len(points), np.inf)
            
        # Get segments: (K, 2, 3)
        segments = self.edge_segments[edge_indices] # K segments
        
        A = segments[:, 0, :]
        B = segments[:, 1, :]
        
        return self._point_segment_distance_batch(points, A, B)
        
    def get_boundary_distance_gpu(self, region_id: int, neighbor_id: int, points_gpu: object, device: str) -> object:
        """
        GPU version of boundary distance.
        points_gpu: (N, 3) torch tensor
        """
        if neighbor_id not in self._boundary_groups.get(region_id, {}):
            return torch.full((len(points_gpu),), float('inf'), device=device)
            
        edge_indices = self._boundary_groups[region_id][neighbor_id]
        if not edge_indices:
            return torch.full((len(points_gpu),), float('inf'), device=device)
        
        key = (region_id, neighbor_id)
        
        # Check cache
        if device not in self.gpu_segments_cache:
            self.gpu_segments_cache[device] = {}
            
        if key not in self.gpu_segments_cache[device]:
            # Upload
            segments_cpu = self.edge_segments[edge_indices] # (K, 2, 3)
            import torch
            A = torch.tensor(segments_cpu[:, 0, :], dtype=torch.float32, device=device)
            B = torch.tensor(segments_cpu[:, 1, :], dtype=torch.float32, device=device)
            self.gpu_segments_cache[device][key] = (A, B)
            
        A, B = self.gpu_segments_cache[device][key]
        
        from ..reconstruction.geometry_gpu import point_segment_distance
        return point_segment_distance(points_gpu, A, B)

    def _point_segment_distance_batch(self, P, A, B):
        """
        P: (N, 3)
        A, B: (K, 3) endpoints of segments
        Returns: (N,) min distance to any segment
        """
        # We need min_k dist(P_i, Seg_k)
        
        # Expand dims:
        # P: (N, 1, 3)
        # A: (1, K, 3)
        # B: (1, K, 3)
        
        P_exp = np.expand_dims(P, 1)
        A_exp = np.expand_dims(A, 0)
        B_exp = np.expand_dims(B, 0)
        
        AB = B_exp - A_exp # (1, K, 3)
        AP = P_exp - A_exp # (N, K, 3)
        
        # Project AP onto AB to find t
        # t = dot(AP, AB) / dot(AB, AB)
        
        AB_sq = np.sum(AB*AB, axis=2) # (1, K)
        AP_dot_AB = np.sum(AP*AB, axis=2) # (N, K)
        
        # Avoid div zero
        t = AP_dot_AB / np.maximum(AB_sq, 1e-12)
        t = np.clip(t, 0.0, 1.0)
        
        # Closest point on segment: C = A + t*AB
        # C dim: (N, K, 3)
        C = A_exp + t[..., np.newaxis] * AB
        
        dist_sq = np.sum((P_exp - C)**2, axis=2) # (N, K)
        
        min_dist_sq = np.min(dist_sq, axis=1) # (N,)
        return np.sqrt(min_dist_sq)

