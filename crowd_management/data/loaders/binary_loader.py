"""
Loader for binary format LiDAR data.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np

from .base_loader import BaseLoader

class BinaryLoader(BaseLoader):
    """Loader for binary format LiDAR data."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the binary loader.
        
        Args:
            config: Optional configuration dictionary with parameters:
                - point_feature_size: Number of features per point (default: 4)
                - dtype: Data type for binary reading (default: np.float32)
        """
        super().__init__(config)
        self.supported_extensions = ['.bin']
        self.point_feature_size = config.get('point_feature_size', 4)
        self.dtype = config.get('dtype', np.float32)
    
    def load(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Load LiDAR data from binary file.
        
        Args:
            file_path: Path to the binary LiDAR data file
            
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
            # Read binary data
            points = np.fromfile(file_path, dtype=self.dtype)
            
            # Reshape to (N, point_feature_size)
            if len(points) % self.point_feature_size != 0:
                raise ValueError(f"Invalid file format: file size is not divisible by {self.point_feature_size}")
                
            points = points.reshape(-1, self.point_feature_size)
            
            # Validate point cloud
            if not self._validate_points(points):
                raise ValueError("Invalid point cloud data")
                
            return points
            
        except Exception as e:
            raise ValueError(f"Error loading binary file: {str(e)}")
    
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
            
        return True 