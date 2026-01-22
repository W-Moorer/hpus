#!/usr/bin/env python
import sys
import os
import numpy as np
import trimesh
import argparse
import torch
from skimage.measure import marching_cubes

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from packages.topology.preprocessing import MeshPreprocessor
from packages.reconstruction.region import RegionReconstructor

def load_custom_mesh(base_path, name):
    """Load mesh from _nodes.txt and _faces.txt or standard mesh file."""
    nodes_file = os.path.join(base_path, f"{name}_nodes.txt")
    faces_file = os.path.join(base_path, f"{name}_faces.txt")
    
    if not os.path.exists(nodes_file) or not os.path.exists(faces_file):
        raise FileNotFoundError(f"Missing {nodes_file} or {faces_file}")
        
    vertices = np.loadtxt(nodes_file)
    faces = np.loadtxt(faces_file, dtype=int)
    
    # Check for normals file
    normals_file = os.path.join(base_path, f"{name}_normals.txt")
    if os.path.exists(normals_file):
        print(f"[Loader] Loading normals from: {normals_file}")
        normals = np.loadtxt(normals_file)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, process=False)
    else:
        print("[Loader] No normals file found, computing normals...")
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.fix_normals()
    
    return mesh

def extract_isosurface(evaluate_func, bounds, resolution=64, level=0.0):
    """Extract isosurface using Marching Cubes with batch evaluation."""
    # Add padding to bounds
    pad = 0.1 * np.ptp(bounds, axis=0).max()
    bmin = bounds[0] - pad
    bmax = bounds[1] + pad
    
    x = np.linspace(bmin[0], bmax[0], resolution)
    y = np.linspace(bmin[1], bmax[1], resolution)
    z = np.linspace(bmin[2], bmax[2], resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    # Batch evaluate to avoid OOM
    batch_size = 50000
    vals = []
    for i in range(0, len(pts), batch_size):
        batch = pts[i:i+batch_size]
        # Ensure batch is correct type/device if needed by evaluate_func
        # But evaluate_func (RegionReconstructor) handles numpy usually
        vals.append(evaluate_func(batch))
    
    F = np.concatenate(vals).reshape(resolution, resolution, resolution)
    
    try:
        verts, faces, normals, values = marching_cubes(F, level=level, spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0]))
        verts += bmin
        return trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    except (ValueError, RuntimeError):
        return None

def main():
    parser = argparse.ArgumentParser(description="Export individual reconstructed regions to subfolders.")
    parser.add_argument("--asset_name", type=str, default="Cube", help="Asset name (e.g. TruncatedRing)")
    parser.add_argument("--asset_dir", type=str, default=None, help="Directory containing asset files")
    parser.add_argument("--output_dir", type=str, default=None, help="Base output directory")
    parser.add_argument("--resolution", type=int, default=64, help="Marching cubes resolution")
    parser.add_argument("--theta0", type=float, default=45.0, help="Sharp edge threshold angle (degrees)")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu, cuda, auto)")
    
    args = parser.parse_args()
    
    # Fix seed for reproducibility
    np.random.seed(42)
    
    # Resolve paths
    if args.asset_dir is None:
        # Default to ../assets/nonsmooth_geometry relative to script
        args.asset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../assets/nonsmooth_geometry'))
    
    if args.output_dir is None:
        # Default to ../output/ply relative to script
        args.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output/ply'))
        
    # Determine device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Config] Using device: {device}")
    
    # Prepare output subdirectory for this model
    model_output_dir = os.path.join(args.output_dir, args.asset_name)
    os.makedirs(model_output_dir, exist_ok=True)
    print(f"[Config] Output directory: {model_output_dir}")
    
    # 1. Load Mesh
    try:
        mesh = load_custom_mesh(args.asset_dir, args.asset_name)
    except Exception as e:
        print(f"[Error] Failed to load mesh: {e}")
        return

    # 2. Preprocess (Partitioning)
    print(f"[Process] Partitioning mesh (Theta0={args.theta0})...")
    preprocessor = MeshPreprocessor(mesh, theta0_degrees=args.theta0)
    topo_data = preprocessor.process()
    regions_indices = topo_data['regions']
    
    print(f"[Process] Found {len(regions_indices)} regions.")
    
    # 3. Reconstruct and Export Each Region
    # Mimic process_assets.py flow:
    # 1. Reconstruct all regions
    # 2. But instead of GlobalImplicitSurface, we export them individually
    
    print("[Process] Reconstructing Regions...")
    for rid in regions_indices:
        print(f"\n--- Processing Region {rid} ---")
        submesh = preprocessor.get_region_mesh(rid)
        
        # Use CPU if requested, but respect args.device
        # process_assets.py uses cpu hardcoded, but we keep flexibility unless strict match is needed
        # User said "extract directly from flow", process_assets.py flow is:
        # recon = RegionReconstructor(rid, submesh, device='cpu')
        # recon.setup_patches()
        # recon.fit()
        # regions.append(recon)
        
        # We will follow this pattern but use the configured device
        recon = RegionReconstructor(rid, submesh, device=device, lambda_penalty=10.0)
        recon.setup_patches()
        recon.fit()
        
        # Extract Isosurface (Logic from process_assets.py's extract step is for Global model)
        # But we need local extraction. 
        # export_regions.py's extract_isosurface function is already good and consistent with fit_region1_hrbf.py
        # We will keep using evaluate_gated as it matches the visual intent of separated regions.
        
        print(f"[Region {rid}] Extracting surface...")
        rec_mesh = extract_isosurface(recon.evaluate_gated, submesh.bounds, resolution=args.resolution)
        
        if rec_mesh and len(rec_mesh.vertices) > 0:
            out_filename = f"region_{rid}.ply"
            out_path = os.path.join(model_output_dir, out_filename)
            rec_mesh.export(out_path)
            print(f"[Region {rid}] Saved to {out_path} ({len(rec_mesh.vertices)} verts)")
        else:
            print(f"[Region {rid}] No valid surface extracted.")

    print("\n[Done] All regions processed.")

if __name__ == "__main__":
    main()
