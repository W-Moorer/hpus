
import sys
import os
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import trimesh
import numpy as np
import argparse

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from packages.topology.preprocessing import MeshPreprocessor
from packages.topology.manager import TopologyManager
from packages.reconstruction.region import RegionReconstructor
from packages.reconstruction.global_model import GlobalImplicitSurface

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, default=None, help='Path to mesh')
    parser.add_argument('--res', type=int, default=100, help='Grid resolution')
    args = parser.parse_args()

    # 1. Load Mesh
    if args.mesh:
        mesh = trimesh.load(args.mesh, force='mesh')
    else:
        print("[Demo] Creating Box Mesh...")
        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))

    # 2. Preprocessing
    print("[Pipeline] Preprocessing Topology...")
    preprocessor = MeshPreprocessor(mesh, theta0_degrees=30.0)
    topo_data = preprocessor.process()
    
    topo_manager = TopologyManager(mesh, topo_data)
    
    # 3. Independent Reconstruction
    print("[Pipeline] Reconstructing Regions...")
    regions = []
    
    # Identify unique region IDs
    unique_rids = list(topo_data['regions'].keys())
    
    for rid in unique_rids:
        # Get submesh
        submesh = preprocessor.get_region_mesh(rid)
        
        # Init Reconstructor
        recon = RegionReconstructor(rid, submesh, device='cpu')
        recon.setup_patches()
        recon.fit() # Fits to submesh surface (implied 0-level set)
        
        regions.append(recon)
        
    # 4. Global Fusion
    print("[Pipeline] building Global Model...")
    global_model = GlobalImplicitSurface(
        regions, 
        topo_manager, 
        epsilon_far=0.001,
        epsilon_edge=0.05, 
        h_bandwidth=0.05,
        lambda_gate=10.0
    )
    
    # 5. Visualization (Slice at Z=0)
    print("[Pipeline] Visualizing Z=0 Slice...")
    x = np.linspace(-0.6, 0.6, args.res)
    y = np.linspace(-0.6, 0.6, args.res)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    
    final_field = global_model.evaluate(pts)
    final_field = final_field.reshape(X.shape)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, final_field, levels=20, cmap='RdBu', vmin=-0.2, vmax=0.2)
    plt.colorbar(label='Field Value')
    plt.contour(X, Y, final_field, levels=[0], colors='black', linewidths=2)
    
    plt.title("Implicit Surface Reconstruction (Z=0 Slice)")
    plt.xlabel("X")
    plt.ylabel("Y")
    output_file = "reconstruction_slice.png"
    plt.savefig(output_file)
    print(f"[Pipeline] Saved visualization to {output_file}")
    
    # Check if edges are rounded
    # Specific check: point (0.5, 0.5) is the corner.
    # The zero level set should slighty curve or pass through?
    # Actually box corner is sharp. Our method produces micro-fillets.
    # So we expect curve connecting x=0.5 and y=0.5 segments.

if __name__ == "__main__":
    main()
