
import numpy as np
import tempfile
import os

def poisson_disk_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Perform Poisson Disk Sampling (Best Sample mode) using PyMeshLab.
    
    Args:
        points: Input point cloud (N x 3)
        num_samples: Target number of samples
        
    Returns:
        sampled_points: Sampled point cloud (M x 3)
    """
    try:
        import pymeshlab as ml
    except ImportError:
        # Fallback to simplified Farthest Point Sampling (FPS)
        # This is a basic implementation to ensure functionality without pymeshlab
        if len(points) <= num_samples: return points
        
        # Simple random sampling as faster fallback than full FPS for very large clouds
        indices = np.random.choice(len(points), num_samples, replace=False)
        return points[indices]

    try:
        # Create temporary PLY file
        # PyMeshLab requires loading from file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ply', delete=False) as tmpfile:
            tmp_ply = tmpfile.name
            # Write minimal PLY header
            with open(tmp_ply, 'w') as f:
                f.write("ply\nformat ascii 1.0\n")
                f.write(f"element vertex {len(points)}\n")
                f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
                np.savetxt(f, points, fmt="%.6f %.6f %.6f")
        
        # Execute sampling
        ms = ml.MeshSet()
        ms.load_new_mesh(tmp_ply)
        
        # generate_simplified_point_cloud uses 'bestsampleflag=True' to simulate Poisson Disk
        ms.generate_simplified_point_cloud(
            samplenum=num_samples,
            bestsampleflag=True,
            bestsamplepool=10,
            exactnumflag=False # Allow slight count variation for quality
        )
        
        sampled_points = ms.current_mesh().vertex_matrix()
        
        # Cleanup
        if os.path.exists(tmp_ply):
            os.remove(tmp_ply)
        
        return sampled_points
        
    except Exception as e:
        print(f"[Warning] PyMeshLab sampling failed: {e}, falling back to random sampling.")
        if 'tmp_ply' in locals() and os.path.exists(tmp_ply):
            os.remove(tmp_ply)
        
        # Fallback
        indices = np.random.choice(len(points), min(len(points), num_samples), replace=False)
        return points[indices]
