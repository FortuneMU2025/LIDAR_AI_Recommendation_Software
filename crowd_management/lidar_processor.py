"""
LiDAR data processing module for crowd management system.
"""

import numpy as np
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import open3d as o3d

class LiDARProcessor:
    def __init__(self, config: Dict):
        """
        Initialize LiDAR processor.
        
        Args:
            config: Configuration dictionary containing processing parameters
        """
        self.config = config
        self.point_feature_size = config.get('point_feature_size', 4)
        self.max_points = config.get('max_points', 100000)
        self.voxel_size = config.get('voxel_size', [0.1, 0.1, 0.1])
        
    def load_data(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Load LiDAR data from file.
        
        Args:
            file_path: Path to LiDAR data file
            
        Returns:
            (N, 4) array of point coordinates and intensities
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"LiDAR file not found: {file_path}")
            
        ext = file_path.suffix.lower()
        
        if ext == '.bin':
            return self._load_bin(file_path)
        elif ext == '.pcd':
            return self._load_pcd(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
    def _load_bin(self, file_path: Path) -> np.ndarray:
        """Load Velodyne binary format."""
        points = np.fromfile(file_path, dtype=np.float32)
        points = points.reshape(-1, 4)  # x, y, z, intensity
        return points
        
    def _load_pcd(self, file_path: Path) -> np.ndarray:
        """Load PCD format."""
        pcd = o3d.io.read_point_cloud(str(file_path))
        points = np.asarray(pcd.points)
        
        if hasattr(pcd, 'colors'):
            # Use first color channel as intensity
            intensity = np.asarray(pcd.colors)[:, 0]
        else:
            # No intensity information
            intensity = np.zeros(len(points))
            
        return np.column_stack([points, intensity])
        
    def preprocess(self, points: np.ndarray) -> np.ndarray:
        """
        Preprocess point cloud data.
        
        Args:
            points: (N, 4) array of point coordinates and intensities
            
        Returns:
            Preprocessed point cloud
        """
        # Remove invalid points
        mask = ~np.isnan(points).any(axis=1)
        points = points[mask]
        
        # Remove points with zero intensity
        mask = points[:, 3] > 0
        points = points[mask]
        
        # Downsample if too many points
        if len(points) > self.max_points:
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]
            
        # Voxelize point cloud
        points = self._voxelize(points)
        
        return points
        
    def _voxelize(self, points: np.ndarray) -> np.ndarray:
        """
        Voxelize point cloud to reduce density.
        
        Args:
            points: (N, 4) array of point coordinates and intensities
            
        Returns:
            Voxelized point cloud
        """
        # Create voxel grid
        voxel_size = np.array(self.voxel_size)
        voxel_grid = np.floor(points[:, :3] / voxel_size).astype(int)
        
        # Get unique voxels
        unique_voxels, inverse_indices = np.unique(voxel_grid, axis=0, return_inverse=True)
        
        # Average points in each voxel
        voxelized_points = np.zeros((len(unique_voxels), 4))
        for i in range(len(unique_voxels)):
            mask = inverse_indices == i
            voxelized_points[i, :3] = np.mean(points[mask, :3], axis=0)
            voxelized_points[i, 3] = np.max(points[mask, 3])  # Use max intensity
            
        return voxelized_points
        
    def transform_coordinates(self, points: np.ndarray, 
                            source_system: str = 'sensor',
                            target_system: str = 'world') -> np.ndarray:
        """
        Transform point cloud coordinates between different coordinate systems.
        
        Args:
            points: (N, 4) array of point coordinates and intensities
            source_system: Source coordinate system ('sensor' or 'world')
            target_system: Target coordinate system ('sensor' or 'world')
            
        Returns:
            Transformed point cloud
        """
        if source_system == target_system:
            return points
            
        # TODO: Implement coordinate transformations
        # This will be expanded based on the specific requirements
        return points 