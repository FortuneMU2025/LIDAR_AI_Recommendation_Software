"""
Tests for the BinaryLoader class.
"""

import numpy as np
import pytest
from pathlib import Path
from crowd_management.data.loaders import BinaryLoader

def test_binary_loader_init(loader_config):
    """Test binary loader initialization."""
    loader = BinaryLoader(loader_config)
    assert loader.point_feature_size == loader_config['point_feature_size']
    assert loader.dtype == loader_config['dtype']
    assert '.bin' in loader.supported_extensions

def test_binary_loader_load(sample_bin_file, sample_point_cloud, loader_config):
    """Test loading binary data."""
    loader = BinaryLoader(loader_config)
    loaded_points = loader.load(sample_bin_file)
    
    assert isinstance(loaded_points, np.ndarray)
    assert loaded_points.shape == sample_point_cloud.shape
    assert np.allclose(loaded_points, sample_point_cloud)

def test_binary_loader_invalid_file(test_data_dir):
    """Test loading invalid binary file."""
    loader = BinaryLoader()
    invalid_file = test_data_dir / "invalid.bin"
    invalid_file.touch()
    
    with pytest.raises(ValueError) as exc_info:
        loader.load(invalid_file)
    assert "Error loading binary file" in str(exc_info.value)

def test_binary_loader_validation(sample_bin_file, loader_config):
    """Test binary loader validation."""
    loader = BinaryLoader(loader_config)
    points = loader.load(sample_bin_file)
    
    # Test validation
    is_valid, error_msg = loader.validate_data(points)
    assert is_valid
    assert error_msg == ""

def test_binary_loader_metadata(sample_bin_file, loader_config):
    """Test binary loader metadata extraction."""
    loader = BinaryLoader(loader_config)
    metadata = loader.get_metadata(sample_bin_file)
    
    assert isinstance(metadata, dict)
    assert 'num_points' in metadata
    assert 'x_range' in metadata
    assert 'y_range' in metadata
    assert 'z_range' in metadata
    assert 'intensity_range' in metadata

def test_binary_loader_custom_dtype(loader_config):
    """Test binary loader with custom data type."""
    custom_config = loader_config.copy()
    custom_config['dtype'] = np.float64
    
    loader = BinaryLoader(custom_config)
    assert loader.dtype == np.float64 