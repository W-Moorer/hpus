
import sys
import os
import trimesh
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from packages.topology.preprocessing import MeshPreprocessor

def test_preprocessing():
    # Create a simple box mesh which has sharp edges
    mesh = trimesh.creation.box(extents=(1,1,1))
    
    # Preprocess
    # Box has 90 degree edges. Theta0 = 30 should detect them.
    preprocessor = MeshPreprocessor(mesh, theta0_degrees=30.0)
    results = preprocessor.process()
    
    print("\n--- Test Results ---")
    print(f"Regions: {len(results['regions'])}")
    print(f"Sharp Edges: {len(results['sharp_edges'])}")
    
    # Verify a box has 6 regions (6 faces)
    if len(results['regions']) == 6:
        print("SUCCESS: 6 regions detected for a box.")
    else:
        print(f"FAILURE: Expected 6 regions, got {len(results['regions'])}.")

    # Verify adjacency
    # Each face should be adjacent to 4 others
    adj = results['region_adjacency']
    for rid in adj.nodes:
        degree = adj.degree[rid]
        print(f"Region {rid} degree: {degree}")
        if degree != 4:
             print(f"WARNING: Region {rid} has degree {degree}, expected 4 for a box.")

    # --- Test TopologyManager ---
    from packages.topology.manager import TopologyManager
    
    manager = TopologyManager(mesh, results)
    
    # Test 1: Closest Region
    # Sample a point on the surface (e.g., center of a face) and add small noise
    # Box is centered at 0? explicit creation says extents=(1,1,1), usually centered at 0.
    test_p = np.array([[0.51, 0, 0], [-0.51, 0, 0]]) # Near +X and -X faces
    dists, pts, rids = manager.get_closest_region(test_p)
    print(f"\n[TopologyManager] Closest Regions for points: {rids}")
    # We expect different regions for opposite sides
    if rids[0] != rids[1]:
        print("SUCCESS: Discrete regions identified for opposite sides.")
    else:
        print("FAILURE: Same region for opposite sides?")

    # Test 2: Boundary Distance
    # A point near the edge between +X and +Y face.
    # Edge at x=0.5, y=0.5, z in [-0.5, 0.5]
    boundary_p = np.array([[0.55, 0.55, 0.0]])
    
    # We need to know which regions are which.
    # Let's just query distance to ALL neighbors of the closest region.
    r0 = manager.get_closest_region(boundary_p)[2][0]
    print(f"Probe point primary region: {r0}")
    
    neighbors = list(manager.region_adjacency.neighbors(r0))
    print(f"Neighbors of {r0}: {neighbors}")
    
    for n_id in neighbors:
        d = manager.get_boundary_distance(r0, n_id, boundary_p)
        print(f"Distance to boundary {r0}->{n_id}: {d[0]:.4f}")
        
    # One of them should be approx sqrt(0.05^2 + 0.05^2) = 0.0707
    # Since point is (0.55, 0.55) and corner is (0.5, 0.5)


if __name__ == "__main__":
    test_preprocessing()
