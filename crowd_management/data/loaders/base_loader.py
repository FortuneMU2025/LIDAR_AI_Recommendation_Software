"""
Base class for LiDAR data loaders.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Union, Tuple
import numpy as np

from ..validators.point_cloud_validator import PointCloudValidator

class BaseLoader(ABC):
    """Abstract base class for all LiDAR data loaders."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the loader.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.supported_extensions = []
        self.validator = PointCloudValidator(config)
        
    @abstractmethod
    def load(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Load LiDAR data from file.
        
        Args:
            file_path: Path to the LiDAR data file
            
        Returns:
            numpy.ndarray: Point cloud data with shape (N, 4) where N is the number of points
                          and 4 represents (x, y, z, intensity)
        """
        pass
    
    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """
        Validate if the file can be loaded by this loader.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            bool: True if file can be loaded, False otherwise
        """
        file_path = Path(file_path)
        return file_path.suffix.lower() in self.supported_extensions
    
    def validate_data(self, points: np.ndarray) -> Tuple[bool, str]:
        """
        Validate point cloud data.
        
        Args:
            points: Point cloud data to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        return self.validator.validate(points)
    
    def get_metadata(self, file_path: Union[str, Path]) -> Dict:
        """
        Get metadata about the point cloud file.
        
        Args:
            file_path: Path to the LiDAR data file
            
        Returns:
            Dict containing metadata about the point cloud
        """
        points = self.load(file_path)
        return self.validator.get_statistics(points) 