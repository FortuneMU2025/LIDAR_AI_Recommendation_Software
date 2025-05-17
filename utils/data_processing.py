import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import re
import io

def load_lidar_data(file_path):
    """
    Load LiDAR point cloud data from a file.
    Supports CSV, XYZ, PLY (simplified), and PCD (simplified) formats.
    
    Args:
        file_path (str): Path to the point cloud file
        
    Returns:
        numpy.ndarray: Points as a numpy array (n, 3)
    """
    try:
        # Detect file format by extension
        file_ext = file_path.lower().split('.')[-1]
        
        if file_ext == 'csv':
            # Assume CSV has headers and x,y,z columns
            data = pd.read_csv(file_path)
            # Try to find x, y, z columns
            xyz_cols = []
            for col in data.columns:
                if col.lower() in ['x', 'y', 'z']:
                    xyz_cols.append(col)
            
            if len(xyz_cols) >= 3:
                # Use the identified columns
                points = data[xyz_cols[:3]].values
            else:
                # Use the first 3 columns
                points = data.iloc[:, :3].values
                
        elif file_ext == 'xyz':
            # Simple space or comma separated XYZ file
            points = np.loadtxt(file_path, delimiter=None)[:, :3]
            
        elif file_ext == 'pcd':
            # Simplified PCD format reader
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Parse header
            header_pattern = re.compile(r'([A-Z_]+)\s+([\w\s\.]+)')
            n_points = None
            data_start = 0
            
            for i, line in enumerate(lines):
                if line.startswith('#') or line.strip() == '':
                    continue
                
                match = header_pattern.match(line.strip())
                if match:
                    if match.group(1) == 'POINTS':
                        n_points = int(match.group(2))
                    continue
                
                # Assume this is where data starts
                data_start = i
                break
            
            # Read points
            points_data = []
            for i in range(data_start, len(lines)):
                line = lines[i].strip()
                if line:
                    values = line.split()
                    if len(values) >= 3:
                        points_data.append([float(val) for val in values[:3]])
            
            points = np.array(points_data)
            
        elif file_ext == 'ply':
            # Simplified PLY format reader
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Parse header
            n_points = None
            data_start = 0
            
            for i, line in enumerate(lines):
                if line.strip() == 'end_header':
                    data_start = i + 1
                    break
                
                if 'element vertex' in line:
                    n_points = int(line.split()[-1])
            
            # Read points
            points_data = []
            for i in range(data_start, data_start + (n_points or len(lines))):
                if i < len(lines):
                    line = lines[i].strip()
                    if line:
                        values = line.split()
                        if len(values) >= 3:
                            points_data.append([float(val) for val in values[:3]])
            
            points = np.array(points_data)
        
        elif file_ext == 'txt':
            # Try to guess format - simple space or comma separated
            points = np.loadtxt(file_path, delimiter=None)[:, :3]
        
        elif file_ext == 'npy':
            # Numpy binary format
            points = np.load(file_path)[:, :3]
            
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Validate points
        if len(points) == 0:
            raise ValueError("The loaded point cloud contains no points")
            
        return points
        
    except Exception as e:
        raise Exception(f"Failed to load point cloud file: {str(e)}")

def preprocess_lidar_data(points):
    """
    Preprocess the LiDAR point cloud data for crowd analysis.
    
    Args:
        points (numpy.ndarray): Array of 3D points
        
    Returns:
        dict: Processed data dictionary containing:
            - points: numpy array of point coordinates (n, 3)
            - colors: numpy array of point colors
            - clusters: numpy array of cluster labels (n,)
            - ground_plane: parameters of the detected ground plane
            - dimensions: dictionary of dataset dimensions
    """
    # Generate default colors based on height (z-value)
    normalized_height = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2]) - np.min(points[:, 2]) + 1e-10)
    colors = np.zeros((len(points), 3))
    colors[:, 0] = normalized_height  # R channel mapped to height
    colors[:, 1] = 0.5 * (1 - normalized_height)  # G channel inverse to height
    colors[:, 2] = 0.5  # B channel constant
    
    # Handle outliers - use statistical approach
    # Compute mean and std along each dimension
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    
    # Filter points within 3 standard deviations of the mean
    mask = np.all(np.abs(points - mean) < 3 * std, axis=1)
    inlier_points = points[mask]
    inlier_colors = colors[mask]
    
    # Create synthetic normals pointing slightly upward
    inlier_normals = np.zeros_like(inlier_points)
    inlier_normals[:, 2] = 1.0  # All normals point up
    
    # Simple ground plane detection - assume lowest 30% of points (by z-value) are ground
    z_threshold = np.percentile(inlier_points[:, 2], 30)
    ground_indices = inlier_points[:, 2] <= z_threshold
    non_ground_indices = ~ground_indices
    
    # Approximate ground plane as z = ax + by + c
    if np.sum(ground_indices) > 10:
        ground_points = inlier_points[ground_indices]
        A = np.column_stack((ground_points[:, 0], ground_points[:, 1], np.ones(len(ground_points))))
        b = ground_points[:, 2]
        # Solve least squares
        try:
            plane_params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            # Convert to ax + by + cz + d = 0 format
            plane_model = np.array([plane_params[0], plane_params[1], -1, plane_params[2]])
        except:
            # Fallback to simple horizontal plane at mean ground height
            plane_model = np.array([0, 0, 1, -np.mean(ground_points[:, 2])])
    else:
        # Fallback if not enough ground points
        plane_model = np.array([0, 0, 1, -np.min(inlier_points[:, 2])])
    
    # Cluster non-ground points to identify potential people using DBSCAN
    non_ground_points = inlier_points[non_ground_indices]
    
    if len(non_ground_points) > 10:
        # Normalize for better clustering
        scaler = StandardScaler()
        scaled_points = scaler.fit_transform(non_ground_points)
        
        # Estimate eps based on data spread
        avg_distance = np.mean(np.std(scaled_points, axis=0)) * 0.5
        eps = max(0.2, min(0.5, avg_distance))
        
        clustering = DBSCAN(eps=eps, min_samples=5).fit(scaled_points)
        cluster_labels = clustering.labels_
    else:
        cluster_labels = np.zeros(len(non_ground_points), dtype=int)
    
    # Map cluster labels back to original points (ground points get label -1)
    full_labels = np.ones(len(inlier_points), dtype=int) * -1
    full_labels[non_ground_indices] = cluster_labels
    
    # Calculate dataset dimensions
    x_min, y_min, z_min = np.min(inlier_points, axis=0)
    x_max, y_max, z_max = np.max(inlier_points, axis=0)
    
    dimensions = {
        'x_range': (x_min, x_max),
        'y_range': (y_min, y_max),
        'z_range': (z_min, z_max),
        'width': x_max - x_min,
        'length': y_max - y_min,
        'height': z_max - z_min
    }
    
    # Create processed data dictionary
    processed_data = {
        'points': inlier_points,
        'colors': inlier_colors,
        'normals': inlier_normals,
        'clusters': full_labels,
        'ground_plane': plane_model,
        'dimensions': dimensions
    }
    
    return processed_data

def downsample_point_cloud(points, factor=0.1):
    """
    Downsample the point cloud for faster visualization or processing.
    
    Args:
        points (numpy.ndarray): Point cloud data as numpy array
        factor (float): Fraction of points to keep (0.0-1.0)
        
    Returns:
        numpy.ndarray: Downsampled point cloud
    """
    if factor >= 1.0:
        return points
    
    num_points = len(points)
    num_keep = max(1, int(num_points * factor))
    indices = np.random.choice(num_points, num_keep, replace=False)
    
    return points[indices]

def extract_people_positions(processed_data):
    """
    Extract estimated positions of people from the clustered point cloud.
    
    Args:
        processed_data (dict): Processed point cloud data
        
    Returns:
        numpy.ndarray: Array of (x, y) positions representing people
    """
    points = processed_data['points']
    clusters = processed_data['clusters']
    
    # Get unique cluster labels, excluding noise (-1)
    unique_clusters = np.unique(clusters)
    unique_clusters = unique_clusters[unique_clusters >= 0]
    
    people_positions = []
    
    for cluster_id in unique_clusters:
        # Get points belonging to this cluster
        cluster_points = points[clusters == cluster_id]
        
        # Use centroid as the person's position
        centroid = np.mean(cluster_points, axis=0)
        
        # Only keep x, y coordinates for 2D position
        people_positions.append(centroid[:2])
    
    return np.array(people_positions)

def calculate_grid_density(people_positions, x_range, y_range, grid_size=1.0):
    """
    Calculate density of people on a grid.
    
    Args:
        people_positions (numpy.ndarray): Array of (x, y) positions of people
        x_range (tuple): (min_x, max_x) range of the area
        y_range (tuple): (min_y, max_y) range of the area
        grid_size (float): Size of each grid cell in meters
        
    Returns:
        tuple: (grid_x, grid_y, density_grid) where grid_x and grid_y are the 
               center coordinates of grid cells, and density_grid is the 2D array 
               of people density per square meter
    """
    if len(people_positions) == 0:
        return None, None, None
    
    # Create grid
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    # Add some margin
    margin = grid_size * 2
    x_min -= margin
    x_max += margin
    y_min -= margin
    y_max += margin
    
    # Create grid cells
    x_edges = np.arange(x_min, x_max + grid_size, grid_size)
    y_edges = np.arange(y_min, y_max + grid_size, grid_size)
    
    # Calculate histograms (density)
    hist, x_edges, y_edges = np.histogram2d(
        people_positions[:, 0], people_positions[:, 1], 
        bins=[x_edges, y_edges]
    )
    
    # Convert counts to density (people per square meter)
    density_grid = hist / (grid_size * grid_size)
    
    # Get grid cell centers
    grid_x = (x_edges[:-1] + x_edges[1:]) / 2
    grid_y = (y_edges[:-1] + y_edges[1:]) / 2
    
    return grid_x, grid_y, density_grid
