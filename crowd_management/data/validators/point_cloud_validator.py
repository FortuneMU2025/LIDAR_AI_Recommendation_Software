"""
Validator for point cloud data.
"""

from typing import Dict, Optional, Tuple
import numpy as np

class PointCloudValidator:
    """Validator for point cloud data."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the validator.
        
        Args:
            config: Optional configuration dictionary with parameters:
                - max_points: Maximum number of points allowed
                - min_points: Minimum number of points required
                - coord_range: Tuple of (min, max) for coordinate values
                - intensity_range: Tuple of (min, max) for intensity values
        """
        config = config or {}
        self.max_points = config.get('max_points', 100000)
        self.min_points = config.get('min_points', 100)
        self.coord_range = config.get('coord_range', (-100, 100))
        self.intensity_range = config.get('intensity_range', (0, 1))
    
    def validate(self, points: np.ndarray) -> Tuple[bool, str]:
        """
        Validate point cloud data.
        
        Args:
            points: Point cloud data to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check shape
        if len(points.shape) != 2 or points.shape[1] != 4:
            return False, "Invalid shape: point cloud must have shape (N, 4)"
            
        # Check number of points
        num_points = len(points)
        if num_points > self.max_points:
            return False, f"too many points: {num_points} > {self.max_points}"
        if num_points < self.min_points:
            return False, f"too few points: {num_points} < {self.min_points}"
            
        # Check for NaN values
        if np.isnan(points).any():
            return False, "Point cloud contains NaN values"
            
        # Check for infinite values
        if np.isinf(points).any():
            return False, "Point cloud contains infinite values"
            
        # Check coordinate ranges
        coords = points[:, :3]
        if (coords < self.coord_range[0]).any() or (coords > self.coord_range[1]).any():
            return False, f"Coordinates out of range: must be between {self.coord_range}"
            
        # Check intensity range
        intensity = points[:, 3]
        if (intensity < self.intensity_range[0]).any() or (intensity > self.intensity_range[1]).any():
            return False, f"Intensity out of range: must be between {self.intensity_range}"
            
        return True, ""
    
    def get_statistics(self, points: np.ndarray) -> Dict:
        """
        Get statistics about the point cloud.
        
        Args:
            points: Point cloud data
            
        Returns:
            Dict containing various statistics
        """
        coords = points[:, :3]
        intensity = points[:, 3]
        
        # Calculate ranges
        x_min, y_min, z_min = coords.min(axis=0)
        x_max, y_max, z_max = coords.max(axis=0)
        
        # Calculate density (points per cubic meter)
        volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        density = len(points) / volume if volume > 0 else 0
        
        return {
            'num_points': len(points),
            'x_range': (float(x_min), float(x_max)),
            'y_range': (float(y_min), float(y_max)),
            'z_range': (float(z_min), float(z_max)),
            'intensity_range': (float(intensity.min()), float(intensity.max())),
            'mean_intensity': float(intensity.mean()),
            'std_intensity': float(intensity.std()),
            'density': float(density)
        } 