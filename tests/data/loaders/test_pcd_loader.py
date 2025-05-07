"""
Tests for the PCDLoader class.
"""

import numpy as np
import pytest
from pathlib import Path
from crowd_management.data.loaders import PCDLoader
from crowd_management.data.loaders.pcd_loader import HAS_OPEN3D

# Skip all tests if Open3D is not available
pytestmark = pytest.mark.skipif(not HAS_OPEN3D, reason="Open3D is not installed")

def test_pcd_loader_init(loader_config):
    """Test PCD loader initialization."""
    loader = PCDLoader(loader_config)
    assert loader.use_intensity == loader_config.get('use_intensity', True)
    assert loader.use_color == loader_config.get('use_color', False)
    assert '.pcd' in loader.supported_extensions

def test_pcd_loader_load(sample_pcd_file, sample_point_cloud, loader_config):
    """Test loading PCD data."""
    loader = PCDLoader(loader_config)
    loaded_points = loader.load(sample_pcd_file)
    
    assert isinstance(loaded_points, np.ndarray)
    assert loaded_points.shape == sample_point_cloud.shape
    assert np.allclose(loaded_points[:, :3], sample_point_cloud[:, :3])  # Check coordinates
    assert np.allclose(loaded_points[:, 3], sample_point_cloud[:, 3])    # Check intensity

def test_pcd_loader_invalid_file(test_data_dir):
    """Test loading invalid PCD file."""
    loader = PCDLoader()
    invalid_file = test_data_dir / "invalid.pcd"
    invalid_file.touch()
    
    with pytest.raises(ValueError) as exc_info:
        loader.load(invalid_file)
    assert "Error loading PCD file" in str(exc_info.value)

def test_pcd_loader_validation(sample_pcd_file, loader_config):
    """Test PCD loader validation."""
    loader = PCDLoader(loader_config)
    points = loader.load(sample_pcd_file)
    
    # Test validation
    is_valid, error_msg = loader.validate_data(points)
    assert is_valid
    assert error_msg == ""

def test_pcd_loader_metadata(sample_pcd_file, loader_config):
    """Test PCD loader metadata extraction."""
    loader = PCDLoader(loader_config)
    metadata = loader.get_metadata(sample_pcd_file)
    
    assert isinstance(metadata, dict)
    assert 'num_points' in metadata
    assert 'x_range' in metadata
    assert 'y_range' in metadata
    assert 'z_range' in metadata
    assert 'intensity_range' in metadata

def test_pcd_loader_without_intensity(loader_config):
    """Test PCD loader without intensity information."""
    custom_config = loader_config.copy()
    custom_config['use_intensity'] = False
    
    loader = PCDLoader(custom_config)
    assert not loader.use_intensity

def test_pcd_loader_with_color(loader_config):
    """Test PCD loader with color information."""
    custom_config = loader_config.copy()
    custom_config['use_color'] = True
    
    loader = PCDLoader(custom_config)
    assert loader.use_color 