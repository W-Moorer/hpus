
import sys
import os
import numpy as np
import trimesh

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from packages.topology.preprocessing import MeshPreprocessor

def load_custom_mesh(base_path, name):
    nodes_file = os.path.join(base_path, f"{name}_nodes.txt")
    faces_file = os.path.join(base_path, f"{name}_faces.txt")
    vertices = np.loadtxt(nodes_file)
    faces = np.loadtxt(faces_file, dtype=int)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.fix_normals()
    return mesh

def main():
    asset_dir = os.path.join(os.path.dirname(__file__), '../assets/nonsmooth_geometry')
    asset_name = "TruncatedRing"
    
    mesh = load_custom_mesh(asset_dir, asset_name)
    print(f"Mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    
    # Use 30 degrees as in the original script, or default 45?
    # User said refer to feature.py which defaults to 45.
    # process_assets.py uses 30.
    # I'll try 30 first (since that gave 20 regions before), then 45 if needed.
    # Actually, if the logic changed, maybe 30 is fine now? 
    # Or maybe the previous logic was just wrong/sensitive.
    # Let's try 30.0 first to see difference with new algorithm.
    
    prep = MeshPreprocessor(mesh, theta0_degrees=30.0)
    res = prep.process()
    print(f"Regions (30 deg): {len(res['regions'])}")
    
    prep45 = MeshPreprocessor(mesh, theta0_degrees=45.0)
    res45 = prep45.process()
    print(f"Regions (45 deg): {len(res45['regions'])}")

if __name__ == "__main__":
    main()
