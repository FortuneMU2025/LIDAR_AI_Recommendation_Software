"""
Tests for the PointCloudValidator class.
"""

import numpy as np
import pytest
from crowd_management.data.validators import PointCloudValidator

def test_validator_init(loader_config):
    """Test validator initialization."""
    validator = PointCloudValidator(loader_config)
    assert validator.max_points == loader_config['max_points']
    assert validator.min_points == loader_config['min_points']
    assert validator.coord_range == loader_config['coord_range']
    assert validator.intensity_range == loader_config['intensity_range']

def test_validate_valid_points(sample_point_cloud, loader_config):
    """Test validation of valid point cloud."""
    validator = PointCloudValidator(loader_config)
    is_valid, error_msg = validator.validate(sample_point_cloud)
    assert is_valid
    assert error_msg == ""

def test_validate_invalid_shape(loader_config):
    """Test validation of invalid point cloud shape."""
    validator = PointCloudValidator(loader_config)
    invalid_points = np.random.rand(100, 3)  # Missing intensity
    is_valid, error_msg = validator.validate(invalid_points)
    assert not is_valid
    assert "Invalid shape" in error_msg

def test_validate_too_many_points(loader_config):
    """Test validation of point cloud with too many points."""
    validator = PointCloudValidator(loader_config)
    too_many_points = np.random.rand(loader_config['max_points'] + 1, 4)
    is_valid, error_msg = validator.validate(too_many_points)
    assert not is_valid
    assert "too many points" in error_msg

def test_validate_too_few_points(loader_config):
    """Test validation of point cloud with too few points."""
    validator = PointCloudValidator(loader_config)
    too_few_points = np.random.rand(loader_config['min_points'] - 1, 4)
    is_valid, error_msg = validator.validate(too_few_points)
    assert not is_valid
    assert "too few points" in error_msg

def test_validate_nan_values(loader_config):
    """Test validation of point cloud with NaN values."""
    validator = PointCloudValidator(loader_config)
    points_with_nan = np.random.rand(1000, 4)
    points_with_nan[0, 0] = np.nan
    is_valid, error_msg = validator.validate(points_with_nan)
    assert not is_valid
    assert "NaN values" in error_msg

def test_validate_infinite_values(loader_config):
    """Test validation of point cloud with infinite values."""
    validator = PointCloudValidator(loader_config)
    points_with_inf = np.random.rand(1000, 4)
    points_with_inf[0, 0] = np.inf
    is_valid, error_msg = validator.validate(points_with_inf)
    assert not is_valid
    assert "infinite values" in error_msg

def test_validate_out_of_range_coords(loader_config):
    """Test validation of point cloud with out-of-range coordinates."""
    validator = PointCloudValidator(loader_config)
    out_of_range_points = np.random.rand(1000, 4)
    out_of_range_points[:, 0] = loader_config['coord_range'][1] + 1
    is_valid, error_msg = validator.validate(out_of_range_points)
    assert not is_valid
    assert "out of range" in error_msg

def test_get_statistics(sample_point_cloud, loader_config):
    """Test getting point cloud statistics."""
    validator = PointCloudValidator(loader_config)
    stats = validator.get_statistics(sample_point_cloud)
    
    assert isinstance(stats, dict)
    assert 'num_points' in stats
    assert 'x_range' in stats
    assert 'y_range' in stats
    assert 'z_range' in stats
    assert 'intensity_range' in stats
    assert 'mean_intensity' in stats
    assert 'std_intensity' in stats
    assert 'density' in stats 