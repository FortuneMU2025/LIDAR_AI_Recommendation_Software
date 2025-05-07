"""
Factory for creating appropriate LiDAR data loaders.
"""

from pathlib import Path
from typing import Dict, Optional, Type

from .base_loader import BaseLoader
from .binary_loader import BinaryLoader
from .pcd_loader import PCDLoader

class LoaderFactory:
    """Factory for creating LiDAR data loaders."""
    
    _loaders: Dict[str, Type[BaseLoader]] = {
        '.bin': BinaryLoader,
        '.pcd': PCDLoader
    }
    
    @classmethod
    def create_loader(cls, file_path: Path, config: Optional[Dict] = None) -> BaseLoader:
        """
        Create appropriate loader for the given file.
        
        Args:
            file_path: Path to the LiDAR data file
            config: Optional configuration dictionary
            
        Returns:
            BaseLoader: Appropriate loader instance
            
        Raises:
            ValueError: If file format is not supported
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension not in cls._loaders:
            supported = ', '.join(cls._loaders.keys())
            raise ValueError(f"Unsupported file format: {extension}. Supported formats: {supported}")
            
        loader_class = cls._loaders[extension]
        return loader_class(config)
    
    @classmethod
    def register_loader(cls, extension: str, loader_class: Type[BaseLoader]) -> None:
        """
        Register a new loader class for a file extension.
        
        Args:
            extension: File extension (e.g., '.las')
            loader_class: Loader class to register
        """
        if not issubclass(loader_class, BaseLoader):
            raise ValueError("Loader class must inherit from BaseLoader")
            
        cls._loaders[extension.lower()] = loader_class
    
    @classmethod
    def get_supported_extensions(cls) -> list:
        """
        Get list of supported file extensions.
        
        Returns:
            list: List of supported file extensions
        """
        return list(cls._loaders.keys()) 