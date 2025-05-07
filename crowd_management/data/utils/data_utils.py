"""
Utility functions for LiDAR data handling.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

def get_file_info(file_path: Union[str, Path]) -> Dict:
    """
    Get information about a LiDAR data file.
    
    Args:
        file_path: Path to the LiDAR data file
        
    Returns:
        Dictionary containing file information
    """
    file_path = Path(file_path)
    return {
        'name': file_path.name,
        'extension': file_path.suffix.lower(),
        'size': file_path.stat().st_size,
        'exists': file_path.exists(),
        'is_file': file_path.is_file()
    }

def list_lidar_files(directory: Union[str, Path], 
                    extensions: Optional[List[str]] = None) -> List[Path]:
    """
    List all LiDAR data files in a directory.
    
    Args:
        directory: Directory to search
        extensions: List of file extensions to include (default: ['.bin', '.pcd'])
        
    Returns:
        List of paths to LiDAR data files
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
        
    if extensions is None:
        extensions = ['.bin', '.pcd']
        
    files = []
    for ext in extensions:
        files.extend(directory.glob(f"*{ext}"))
        
    return sorted(files)

def normalize_intensity(points: np.ndarray, 
                       target_range: tuple = (0, 1)) -> np.ndarray:
    """
    Normalize intensity values to target range.
    
    Args:
        points: Point cloud data with shape (N, 4)
        target_range: Target range for intensity values
        
    Returns:
        Point cloud with normalized intensity values
    """
    points = points.copy()
    min_val, max_val = target_range
    current_min = points[:, 3].min()
    current_max = points[:, 3].max()
    
    if current_max > current_min:
        points[:, 3] = (points[:, 3] - current_min) / (current_max - current_min)
        points[:, 3] = points[:, 3] * (max_val - min_val) + min_val
        
    return points

def remove_outliers(points: np.ndarray, 
                   distance_threshold: float = 2.0,
                   min_neighbors: int = 3) -> np.ndarray:
    """
    Remove outlier points based on distance to neighbors.
    
    Args:
        points: Point cloud data with shape (N, 4)
        distance_threshold: Maximum distance to consider points as neighbors
        min_neighbors: Minimum number of neighbors required to keep a point
        
    Returns:
        Point cloud with outliers removed
    """
    from scipy.spatial import KDTree
    
    # Build KD-tree for efficient neighbor search
    tree = KDTree(points[:, :3])
    
    # Find neighbors for each point
    neighbors = tree.query_ball_point(points[:, :3], distance_threshold)
    
    # Keep points with enough neighbors
    mask = np.array([len(n) >= min_neighbors for n in neighbors])
    
    return points[mask] 