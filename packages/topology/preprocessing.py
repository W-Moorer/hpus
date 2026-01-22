
import trimesh
import numpy as np
import networkx as nx
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional, Any

# --- Helper functions ported from feature.py ---

def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    normals = np.cross(e1, e2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return normals / norms

def build_edge_face_map(faces: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    edge_face_map = defaultdict(list)
    for face_idx, face in enumerate(faces):
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            edge = (min(v1, v2), max(v1, v2))
            edge_face_map[edge].append(face_idx)
    return dict(edge_face_map)

def detect_sharp_edges(vertices: np.ndarray, faces: np.ndarray, theta0_deg: float) -> Set[Tuple[int, int]]:
    face_normals = compute_face_normals(vertices, faces)
    edge_face_map = build_edge_face_map(faces)
    theta0_rad = np.radians(theta0_deg)
    sharp_edges = set()
    
    for edge, face_indices in edge_face_map.items():
        if len(face_indices) != 2:
            # Boundary or non-manifold -> sharp
            sharp_edges.add(edge)
            continue
        
        f1, f2 = face_indices
        n1, n2 = face_normals[f1], face_normals[f2]
        cos_theta = np.clip(np.dot(n1, n2), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        if theta >= theta0_rad:
            sharp_edges.add(edge)
            
    return sharp_edges

def segment_smooth_regions(faces: np.ndarray, sharp_edges: Set[Tuple[int, int]]) -> List[List[int]]:
    n_faces = len(faces)
    edge_face_map = build_edge_face_map(faces)
    
    # Adjacency avoiding sharp edges
    adjacency = defaultdict(set)
    for edge, face_indices in edge_face_map.items():
        if len(face_indices) == 2 and edge not in sharp_edges:
            f1, f2 = face_indices
            adjacency[f1].add(f2)
            adjacency[f2].add(f1)
            
    visited = np.zeros(n_faces, dtype=bool)
    regions = []
    
    for start_face in range(n_faces):
        if visited[start_face]:
            continue
            
        region = []
        queue = [start_face]
        visited[start_face] = True
        
        while queue:
            face = queue.pop(0)
            region.append(face)
            
            for neighbor in adjacency[face]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        regions.append(region)
        
    return regions

class MeshPreprocessor:
    """
    Preprocesses a mesh for Partitioned HRBF-PoU reconstruction.
    
    Key responsibility:
    1. Detect sharp edges based on dihedral angle using MANUAL calculation (robust).
    2. Partition mesh into smooth regions using BFS on safe edges.
    3. Build region adjacency graph.
    """
    
    def __init__(self, mesh_path_or_obj, theta0_degrees: float = 45.0):
        if isinstance(mesh_path_or_obj, (str, Path)):
            self.mesh = trimesh.load(mesh_path_or_obj, force='mesh')
        else:
            self.mesh = mesh_path_or_obj
            
        self.theta0 = theta0_degrees
        
        # Results
        self.sharp_edges: Set[Tuple[int, int]] = set() 
        self.regions: List[List[int]] = [] # Region ID -> List of Face Indices
        self.region_adjacency: nx.Graph = nx.Graph()
        self.face_region_map: np.ndarray = None 
        
        # Derived sub-meshes
        self.region_meshes: Dict[int, trimesh.Trimesh] = {}

    def process(self):
        """Run the full preprocessing pipeline."""
        print("[MeshPreprocessor] Detecting sharp edges...")
        self.sharp_edges = detect_sharp_edges(self.mesh.vertices, self.mesh.faces, self.theta0)
        print(f"[MeshPreprocessor] Found {len(self.sharp_edges)} sharp edges (Threshold: {self.theta0} deg).")
        
        print("[MeshPreprocessor] Partitioning regions...")
        self.regions = segment_smooth_regions(self.mesh.faces, self.sharp_edges)
        
        self.face_region_map = np.full(len(self.mesh.faces), -1)
        for rid, faces in enumerate(self.regions):
            self.face_region_map[faces] = rid
            
        print(f"[MeshPreprocessor] Partitioned into {len(self.regions)} smooth regions.")
        
        print("[MeshPreprocessor] Building adjacency graph...")
        self._build_region_adjacency()
        
        return {
            'regions': {i: f for i, f in enumerate(self.regions)},
            'region_adjacency': self.region_adjacency,
            'face_region_map': self.face_region_map,
            'sharp_edges': self.sharp_edges
        }

    def _build_region_adjacency(self):
        """Build region connectivity graph based on shared sharp edges."""
        self.region_adjacency = nx.Graph()
        self.region_adjacency.add_nodes_from(range(len(self.regions)))
        
        edge_face_map = build_edge_face_map(self.mesh.faces)
        
        for edge in self.sharp_edges:
            faces = edge_face_map.get(edge)
            if not faces or len(faces) < 2:
                continue
                
            rids = set()
            for f in faces:
                rid = self.face_region_map[f]
                if rid != -1:
                    rids.add(rid)
            
            if len(rids) == 2:
                u, v = list(rids)
                self.region_adjacency.add_edge(u, v)

    def get_region_mesh(self, region_id: int) -> trimesh.Trimesh:
        """Extract a submesh for a specific region."""
        if region_id in self.region_meshes:
            return self.region_meshes[region_id]
            
        if region_id < 0 or region_id >= len(self.regions):
            raise ValueError(f"Invalid region ID {region_id}")
        
        face_indices = self.regions[region_id]
        submesh = self.mesh.submesh([face_indices], append=True)
        self.region_meshes[region_id] = submesh
        return submesh

    def export_regions(self, output_dir: str):
        """Save each region as a separate OBJ/PLY."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        for rid in range(len(self.regions)):
            sub = self.get_region_mesh(rid)
            sub.export(os.path.join(output_dir, f"region_{rid}.obj"))
