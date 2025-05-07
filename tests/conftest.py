"""
Pytest configuration file with common fixtures.
"""

import os
import numpy as np
import pytest
from pathlib import Path

@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    return tmp_path

@pytest.fixture
def sample_point_cloud():
    """Create a sample point cloud for testing."""
    num_points = 1000
    # Generate points within valid ranges
    coords = np.random.uniform(-100, 100, (num_points, 3))
    intensity = np.random.uniform(0, 1, (num_points, 1))
    points = np.hstack([coords, intensity])
    return points.astype(np.float32)  # Ensure correct dtype

@pytest.fixture
def sample_bin_file(test_data_dir, sample_point_cloud):
    """Create a sample binary file for testing."""
    file_path = test_data_dir / "test.bin"
    sample_point_cloud.astype(np.float32).tofile(file_path)  # Ensure correct dtype
    return file_path

@pytest.fixture
def sample_pcd_file(test_data_dir, sample_point_cloud):
    """Create a sample PCD file for testing."""
    import open3d as o3d
    file_path = test_data_dir / "test.pcd"
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sample_point_cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(
        np.column_stack([sample_point_cloud[:, 3], np.zeros((len(sample_point_cloud), 2))])
    )
    
    # Save to file
    o3d.io.write_point_cloud(str(file_path), pcd)
    return file_path

@pytest.fixture
def loader_config():
    """Create a sample loader configuration."""
    return {
        'max_points': 100000,
        'min_points': 100,
        'coord_range': (-100, 100),
        'intensity_range': (0, 1),
        'point_feature_size': 4,
        'dtype': np.float32
    } 