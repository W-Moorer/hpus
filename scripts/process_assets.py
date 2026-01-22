
import sys
import os
import glob
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.measure import marching_cubes
import argparse

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from packages.topology.preprocessing import MeshPreprocessor
from packages.topology.manager import TopologyManager
from packages.reconstruction.region import RegionReconstructor
from packages.reconstruction.global_model import GlobalImplicitSurface

def load_custom_mesh(base_path, name):
    """Load mesh from _nodes.txt and _faces.txt"""
    nodes_file = os.path.join(base_path, f"{name}_nodes.txt")
    faces_file = os.path.join(base_path, f"{name}_faces.txt")
    
    if not os.path.exists(nodes_file) or not os.path.exists(faces_file):
        raise FileNotFoundError(f"Missing {nodes_file} or {faces_file}")
        
    vertices = np.loadtxt(nodes_file)
    faces = np.loadtxt(faces_file, dtype=int)
    
    # Create Trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.fix_normals()
    return mesh

def plot_4view(mesh, output_path):
    """Generate 4-view visualization using matplotlib"""
    fig = plt.figure(figsize=(12, 12))
    
    views = [
        (221, 'Front (XZ)', (0, 90)),
        (222, 'Side (YZ)', (0, 0)),
        (223, 'Top (XY)', (90, -90)),
        (224, 'Isometric', (30, 45))
    ]
    
    # Downsample for plotting if heavy
    # if len(mesh.vertices) > 10000:
    #     plot_mesh = mesh.simplify_quadric_decimation(5000)
    # else:
    plot_mesh = mesh
        
    v = plot_mesh.vertices
    f = plot_mesh.faces
    
    for subplot, title, elev_azim in views:
        ax = fig.add_subplot(subplot, projection='3d')
        ax.set_title(title)
        
        # Plot trisurf
        ax.plot_trisurf(v[:,0], v[:,1], v[:,2], triangles=f, color='lightblue', alpha=0.8, edgecolor='k', linewidth=0.1, shade=True)
        
        # Set aspect equal
        mid = (np.max(v, 0) + np.min(v, 0)) / 2
        rng = np.max(np.ptp(v, 0)) / 2
        ax.set_xlim(mid[0]-rng, mid[0]+rng)
        ax.set_ylim(mid[1]-rng, mid[1]+rng)
        ax.set_zlim(mid[2]-rng, mid[2]+rng)
        
        ax.view_init(elev=elev_azim[0], azim=elev_azim[1])
        ax.set_axis_off()
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()

def main():
    asset_dir = os.path.join(os.path.dirname(__file__), '../assets/nonsmooth_geometry')
    asset_name = "TruncatedRing"
    
    print(f"[Process] Loading {asset_name} from {asset_dir}...")
    mesh = load_custom_mesh(asset_dir, asset_name)
    print(f"[Process] Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces.")
    
    # 1. Pipeline
    print("[Process] Preprocessing...")
    preprocessor = MeshPreprocessor(mesh, theta0_degrees=45.0)
    topo_data = preprocessor.process()
    topo_manager = TopologyManager(mesh, topo_data)
    
    regions = []
    print("[Process] Reconstructing Regions...")
    for rid in topo_data['regions']:
        submesh = preprocessor.get_region_mesh(rid)
        recon = RegionReconstructor(rid, submesh, device='cpu') # Use CPU for safety
        recon.setup_patches()
        recon.fit()
        regions.append(recon)
        
    global_model = GlobalImplicitSurface(regions, topo_manager, 
                                         device='cpu',
                                         epsilon_far=1e-3, # Tuned for CAD scale ~1.0
                                         epsilon_edge=0.02, 
                                         h_bandwidth=0.05)
                                         
    # Set Times New Roman font
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    
    # Ensure figures dir exists
    figures_dir = os.path.join(os.path.dirname(__file__), '../figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Ensure output dir exists
    output_dir = os.path.join(os.path.dirname(__file__), '../output')
    os.makedirs(output_dir, exist_ok=True)
    
    output_ply = os.path.join(output_dir, "TruncatedRing_reconstructed.ply")
    
    if os.path.exists(output_ply):
        print(f"[Process] Found existing {output_ply}. Skipping reconstruction.")
        rec_mesh = trimesh.load(output_ply)
        # Verify it loaded as mesh
        if isinstance(rec_mesh, trimesh.Scene):
             if len(rec_mesh.geometry) > 0:
                 rec_mesh = list(rec_mesh.geometry.values())[0]
             else:
                 print("Error loading PLY as scene.")
                 
        output_png = os.path.join(figures_dir, "TruncatedRing_4view.png")
        plot_4view(rec_mesh, output_png)
        print(f"[Process] Saved PNG to {output_png}")
        return

    print("[Process] Extracting Iso-surface...")
    # Bounding box + padding
    bounds = mesh.bounds
    pad = 0.1 * np.ptp(bounds, axis=0).max()
    bmin = bounds[0] - pad
    bmax = bounds[1] + pad
    
    res = 60
    x = np.linspace(bmin[0], bmax[0], res)
    y = np.linspace(bmin[1], bmax[1], res)
    z = np.linspace(bmin[2], bmax[2], res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    # Batch evaluate to avoid mem OOM? 100^3 = 1M points. OK for standard RAM.
    # But let's batch just in case.
    batch_size = 100000
    vals = []
    for i in range(0, len(pts), batch_size):
        batch = pts[i:i+batch_size]
        vals.append(global_model.evaluate(batch))
    
    F = np.concatenate(vals).reshape(res, res, res)
    
    # Extract
    try:
        verts, faces, normals, values = marching_cubes(F, level=0.0, spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0]))
        # Shift vertices to world space
        verts += bmin
        
        rec_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
        
        # Save PLY
        rec_mesh.export(output_ply)
        print(f"[Process] Saved PLY to {output_ply}")
        
        # Save 4-view
        output_png = os.path.join(figures_dir, "TruncatedRing_4view.png")
        plot_4view(rec_mesh, output_png)
        print(f"[Process] Saved PNG to {output_png}")
        
    except ValueError:
        print("[Process] Marching Cubes failed (no zero level set found?).")

if __name__ == "__main__":
    main()
