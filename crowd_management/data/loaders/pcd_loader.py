"""
Loader for PCD format LiDAR data.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import open3d as o3d

from .base_loader import BaseLoader

class PCDLoader(BaseLoader):
    """Loader for PCD format LiDAR data."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the PCD loader.
        
        Args:
            config: Optional configuration dictionary with parameters:
                - use_intensity: Whether to use intensity from PCD (default: True)
                - use_color: Whether to use color information (default: False)
        """
        super().__init__(config)
        self.supported_extensions = ['.pcd']
        self.use_intensity = config.get('use_intensity', True)
        self.use_color = config.get('use_color', False)
    
    def load(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Load LiDAR data from PCD file.
        
        Args:
            file_path: Path to the PCD LiDAR data file
            
        Returns:
            numpy.ndarray: Point cloud data with shape (N, 4)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"LiDAR file not found: {file_path}")
            
        try:
            # Read PCD file
            pcd = o3d.io.read_point_cloud(str(file_path))
            
            # Get points
            points = np.asarray(pcd.points)
            
            # Get intensity/color information
            if self.use_intensity and hasattr(pcd, 'intensity'):
                intensity = np.asarray(pcd.intensity)
            elif self.use_color and hasattr(pcd, 'colors'):
                # Use first color channel as intensity
                intensity = np.asarray(pcd.colors)[:, 0]
            else:
                # No intensity information available
                intensity = np.zeros(len(points))
            
            # Combine points and intensity
            point_cloud = np.column_stack([points, intensity])
            
            # Validate point cloud
            if not self._validate_points(point_cloud):
                raise ValueError("Invalid point cloud data")
                
            return point_cloud
            
        except Exception as e:
            raise ValueError(f"Error loading PCD file: {str(e)}")
    
    def _validate_points(self, points: np.ndarray) -> bool:
        """
        Validate point cloud data.
        
        Args:
            points: Point cloud data to validate
            
        Returns:
            bool: True if points are valid, False otherwise
        """
        # Check for NaN values
        if np.isnan(points).any():
            return False
            
        # Check for infinite values
        if np.isinf(points).any():
            return False
            
        # Check coordinate ranges (typical LiDAR ranges)
        if (points[:, :3] > 1000).any() or (points[:, :3] < -1000).any():
            return False
            
        # Check intensity range
        if (points[:, 3] < 0).any() or (points[:, 3] > 1).any():
            return False
            
        return True 