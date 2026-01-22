
import sys
import os
import trimesh
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from packages.reconstruction.region import RegionReconstructor

def test_region_reconstruction():
    print("--- Test Region Reconstruction ---")
    
    # Create a sphere (smooth region)
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Initialize Reconstructor
    # Use CPU for test to avoid CUDA requirement on CI/agents without GPU
    device = 'cpu' 
    recon = RegionReconstructor(region_id=0, mesh=mesh, device=device)
    
    # 1. Setup Patches
    recon.setup_patches()
    if recon.patch_centers is None:
        print("FAILURE: Patch setup failed.")
        return
    
    # 2. Fit
    recon.fit()
    if len(recon.solvers) == 0:
        print("FAILURE: Fitting failed (no solvers).")
        return
        
    # 3. Evaluate
    # Query points on surface
    sample_p = mesh.vertices[:5]
    vals = recon.evaluate(sample_p)
    
    print(f"Values on surface (expect ~0): {vals}")
    mae = np.mean(np.abs(vals))
    print(f"MAE on surface: {mae:.6f}")
    
    if mae < 1e-2:
        print("SUCCESS: Reconstruction fits surface.")
    else:
        print("WARNING: Fitting error might be too high?")

    # 4. Gated Evaluate
    # Query far point
    far_p = np.array([[10.0, 0.0, 0.0]])
    gated_val = recon.evaluate_gated(far_p)
    print(f"Gated value at distance 10 (expect large): {gated_val}")
    
    if gated_val > 5.0:
         print("SUCCESS: Gating works (high penalty far away).")
    else:
         print("FAILURE: Gating penalty too low?")

if __name__ == "__main__":
    test_region_reconstruction()
