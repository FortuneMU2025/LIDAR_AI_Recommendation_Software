"""
Configuration settings for LiDAR processing.
"""

import os
from pathlib import Path

# Default configuration
DEFAULT_CONFIG = {
    # LiDAR settings
    'point_feature_size': 4,  # x, y, z, intensity
    'max_points': 100000,
    'voxel_size': [0.1, 0.1, 0.1],
    
    # Output settings
    'output_dir': str(Path.home() / '.lidar_processing/output'),
    'save_visualizations': True
}

# Create default directories
os.makedirs(DEFAULT_CONFIG['output_dir'], exist_ok=True)

"""
Configuration for NuScenes integration.
"""

class NuScenesConfig:
    """Configuration for NuScenes data processing."""
    
    def __init__(self):
        self.lidar_config = {
            'min_points': 100,
            'max_points': 100000,
            'voxel_size': 0.1
        }
        self.coord_system = {
            'source': 'nuscenes',
            'target': 'world'
        }
        self.batch_size = 32
        self.visualization = {
            'save_detections': True,
            'save_density': True,
            'save_flow': True,
            'save_groups': True,
            'save_bottlenecks': True
        } 