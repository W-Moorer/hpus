
import sys
import os
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Configure Matplotlib to use Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  # Better math font matching Times

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from packages.topology.preprocessing import MeshPreprocessor

def plot_region_analysis(region_id, mesh, output_path):
    """Generate 4-view visualization for a region including nodes, topology, and normals."""
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Region {region_id} Analysis (Nodes, Topology, Normals)", fontsize=16)
    
    # Views: Elev, Azim
    views = [
        (221, 'Front (XZ)', (0, 90)),
        (222, 'Side (YZ)', (0, 0)),
        (223, 'Top (XY)', (90, -90)),
        (224, 'Isometric', (30, 45))
    ]
    
    v = mesh.vertices
    f = mesh.faces
    n = mesh.vertex_normals
    
    # Calculate bounds for consistent axis
    if len(v) > 0:
        mid = (np.max(v, 0) + np.min(v, 0)) / 2
        max_range = np.max(np.ptp(v, 0)) / 2
    else:
        mid = np.zeros(3)
        max_range = 1.0

    # Ensure a minimum range to avoid singular transformation if flat
    if max_range < 1e-6:
        max_range = 1.0

    for subplot, title, elev_azim in views:
        ax = fig.add_subplot(subplot, projection='3d')
        ax.set_title(title)
        
        # 1. Topology (Wireframe) - Plot edges
        # Use shade=False and explicit edge colors to ensure visibility
        # Face color is set to fully transparent
        ax.plot_trisurf(v[:,0], v[:,1], v[:,2], triangles=f, 
                       color=(1, 1, 1, 0.0), 
                       edgecolor='black', 
                       linewidth=0.8, 
                       shade=False,
                       antialiased=True,
                       label='Topology')
        
        # 2. Nodes (Scatter)
        ax.scatter(v[:,0], v[:,1], v[:,2], c='blue', s=20, depthshade=False, label='Nodes')
        
        # 3. Normals (Quiver)
        # Scale normals for visibility (e.g., 20% of range)
        scale = max_range * 0.2
        # Downsample normals if too many points to avoid clutter, but for submesh it should be fine
        ax.quiver(v[:,0], v[:,1], v[:,2], n[:,0], n[:,1], n[:,2], length=scale, color='red', linewidth=1.0, label='Normals')
        
        # Setup axes
        ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
        ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
        ax.set_zlim(mid[2]-max_range, mid[2]+max_range)
        
        # Aspect ratio hack
        try:
            ax.set_box_aspect([1, 1, 1])
        except AttributeError:
            pass
            
        ax.view_init(elev=elev_azim[0], azim=elev_azim[1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    # Legend (only on last plot to save space, or global)
    # ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"Saved visualization to {output_path}")

def main():
    print("Loading Cube...")
    base_path = "e:/workspace/hpus/assets/nonsmooth_geometry/"
    node_path = base_path + "Cube_nodes.txt"
    face_path = base_path + "Cube_faces.txt"
    norm_path = base_path + "Cube_normals.txt"

    if os.path.exists(node_path) and os.path.exists(face_path):
        print(f"[Loader] Loading from txt files: {node_path}")
        vertices = np.loadtxt(node_path)
        faces = np.loadtxt(face_path, dtype=int)
        
        if os.path.exists(norm_path):
            print(f"[Loader] Loading normals from: {norm_path}")
            normals = np.loadtxt(norm_path)
            # Verify shapes
            if len(normals) != len(vertices):
                print(f"Warning: Normals count {len(normals)} != Vertices count {len(vertices)}")
            
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals, process=False)
        else:
            print("Normals file not found, auto-computing.")
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            mesh.fix_normals()
    else:
        print("Error: Input files not found.")
        return

    print("Partitioning...")
    # Using 45 degrees as in previous steps
    preprocessor = MeshPreprocessor(mesh, theta0_degrees=45.0)
    preprocessor.process()
    
    output_dir = "e:/workspace/hpus/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing {len(preprocessor.regions)} regions...")
    for rid in range(len(preprocessor.regions)):
        print(f"Visualizing Region {rid}...")
        submesh = preprocessor.get_region_mesh(rid)
        output_file = os.path.join(output_dir, f"region_{rid}_analysis.png")
        plot_region_analysis(rid, submesh, output_file)
    
    print("Done.")

if __name__ == "__main__":
    main()
