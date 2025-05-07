"""
Tests for the LoaderFactory class.
"""

import pytest
from pathlib import Path
from crowd_management.data.loaders import LoaderFactory, BaseLoader, BinaryLoader, PCDLoader
from crowd_management.data.loaders.pcd_loader import HAS_OPEN3D

def test_create_loader_binary(sample_bin_file, loader_config):
    """Test creating a binary loader."""
    loader = LoaderFactory.create_loader(sample_bin_file, loader_config)
    assert isinstance(loader, BinaryLoader)
    assert loader.config == loader_config

@pytest.mark.skipif(not HAS_OPEN3D, reason="Open3D is not installed")
def test_create_loader_pcd(sample_pcd_file, loader_config):
    """Test creating a PCD loader."""
    loader = LoaderFactory.create_loader(sample_pcd_file, loader_config)
    assert isinstance(loader, PCDLoader)
    assert loader.config == loader_config

def test_create_loader_invalid_format(test_data_dir):
    """Test creating a loader for invalid format."""
    invalid_file = test_data_dir / "test.invalid"
    invalid_file.touch()
    
    with pytest.raises(ValueError) as exc_info:
        LoaderFactory.create_loader(invalid_file)
    assert "Unsupported file format" in str(exc_info.value)

def test_register_loader():
    """Test registering a new loader."""
    class CustomLoader(BaseLoader):
        def load(self, file_path):
            return None
    
    # Register new loader
    LoaderFactory.register_loader('.custom', CustomLoader)
    
    # Verify registration
    assert '.custom' in LoaderFactory.get_supported_extensions()
    
    # Test creating custom loader
    loader = LoaderFactory.create_loader(Path('test.custom'))
    assert isinstance(loader, CustomLoader)

def test_register_invalid_loader():
    """Test registering an invalid loader."""
    class InvalidLoader:
        pass
    
    with pytest.raises(ValueError) as exc_info:
        LoaderFactory.register_loader('.invalid', InvalidLoader)
    assert "must inherit from BaseLoader" in str(exc_info.value)

def test_get_supported_extensions():
    """Test getting supported extensions."""
    extensions = LoaderFactory.get_supported_extensions()
    assert '.bin' in extensions
    if HAS_OPEN3D:
        assert '.pcd' in extensions
    assert isinstance(extensions, list) 