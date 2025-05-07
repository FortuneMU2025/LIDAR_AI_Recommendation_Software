"""
Point cloud data validation module.
"""

from typing import Dict, Tuple
import numpy as np

class PointCloudValidator:
    """Validator for point cloud data."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize validator.
        
        Args:
            config: Configuration dictionary with validation parameters:
                - max_points: Maximum number of points allowed
                - min_points: Minimum number of points required
                - coord_range: Tuple of (min, max) for coordinate values
                - intensity_range: Tuple of (min, max) for intensity values
        """
        self.config = config or {}
        self.max_points = self.config.get('max_points', 1000000)
        self.min_points = self.config.get('min_points', 100)
        self.coord_range = self.config.get('coord_range', (-1000, 1000))
        self.intensity_range = self.config.get('intensity_range', (0, 1))
    
    def validate(self, points: np.ndarray) -> Tuple[bool, str]:
        """
        Validate point cloud data.
        
        Args:
            points: Point cloud data with shape (N, 4)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check shape
        if len(points.shape) != 2 or points.shape[1] != 4:
            return False, f"Invalid shape: expected (N, 4), got {points.shape}"
            
        # Check number of points
        if len(points) > self.max_points:
            return False, f"Too many points: {len(points)} > {self.max_points}"
        if len(points) < self.min_points:
            return False, f"Too few points: {len(points)} < {self.min_points}"
            
        # Check for NaN values
        if np.isnan(points).any():
            return False, "Point cloud contains NaN values"
            
        # Check for infinite values
        if np.isinf(points).any():
            return False, "Point cloud contains infinite values"
            
        # Check coordinate ranges
        coord_min, coord_max = self.coord_range
        if (points[:, :3] < coord_min).any() or (points[:, :3] > coord_max).any():
            return False, f"Coordinates outside valid range [{coord_min}, {coord_max}]"
            
        # Check intensity range
        intensity_min, intensity_max = self.intensity_range
        if (points[:, 3] < intensity_min).any() or (points[:, 3] > intensity_max).any():
            return False, f"Intensities outside valid range [{intensity_min}, {intensity_max}]"
            
        return True, "Point cloud is valid"
    
    def get_statistics(self, points: np.ndarray) -> Dict:
        """
        Get statistics about the point cloud.
        
        Args:
            points: Point cloud data with shape (N, 4)
            
        Returns:
            Dictionary containing point cloud statistics
        """
        return {
            'num_points': len(points),
            'x_range': (points[:, 0].min(), points[:, 0].max()),
            'y_range': (points[:, 1].min(), points[:, 1].max()),
            'z_range': (points[:, 2].min(), points[:, 2].max()),
            'intensity_range': (points[:, 3].min(), points[:, 3].max()),
            'mean_intensity': points[:, 3].mean(),
            'std_intensity': points[:, 3].std(),
            'density': len(points) / (points[:, :3].max(axis=0) - points[:, :3].min(axis=0)).prod()
        } 