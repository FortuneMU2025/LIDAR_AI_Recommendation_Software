"""
LiDAR data processing module for crowd management system.
"""

import numpy as np
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import open3d as o3d
import logging
import time

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LiDARProcessor:
    def __init__(self, config: Dict):
        """
        Initialize LiDAR processor.
        
        Args:
            config: Configuration dictionary containing processing parameters
        """
        self.config = config
        self.point_feature_size = config.get('point_feature_size', 4)
        self.max_points = config.get('max_points', 100000)
        self.voxel_size = config.get('voxel_size', [0.1, 0.1, 0.1])
        logger.info(f"Initialized LiDARProcessor with config: {config}")
        
    def load_data(self, file_path: Union[str, Path]) -> np.ndarray:
        """
        Load LiDAR data from file.
        
        Args:
            file_path: Path to LiDAR data file
            
        Returns:
            (N, 4) array of point coordinates and intensities
        """
        start_time = time.time()
        file_path = Path(file_path)
        logger.info(f"Loading data from: {file_path}")
        
        if not file_path.exists():
            logger.error(f"LiDAR file not found: {file_path}")
            raise FileNotFoundError(f"LiDAR file not found: {file_path}")
            
        ext = file_path.suffix.lower()
        logger.info(f"File format: {ext}")
        
        if ext == '.bin':
            points = self._load_bin(file_path)
        elif ext == '.pcd':
            points = self._load_pcd(file_path)
        else:
            logger.error(f"Unsupported file format: {ext}")
            raise ValueError(f"Unsupported file format: {ext}")
            
        load_time = time.time() - start_time
        logger.info(f"Loaded {len(points)} points in {load_time:.2f} seconds")
        logger.info(f"Point cloud statistics:")
        logger.info(f"  - Shape: {points.shape}")
        logger.info(f"  - Coordinate range: X[{points[:,0].min():.2f}, {points[:,0].max():.2f}], "
                   f"Y[{points[:,1].min():.2f}, {points[:,1].max():.2f}], "
                   f"Z[{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
        logger.info(f"  - Intensity range: [{points[:,3].min():.2f}, {points[:,3].max():.2f}]")
        
        return points
        
    def _load_bin(self, file_path: Path) -> np.ndarray:
        """Load Velodyne binary format."""
        logger.debug("Loading binary format")
        points = np.fromfile(file_path, dtype=np.float32)
        points = points.reshape(-1, 4)  # x, y, z, intensity
        return points
        
    def _load_pcd(self, file_path: Path) -> np.ndarray:
        """Load PCD format."""
        logger.debug("Loading PCD format")
        pcd = o3d.io.read_point_cloud(str(file_path))
        points = np.asarray(pcd.points)
        
        if hasattr(pcd, 'colors'):
            logger.debug("Using color information for intensity")
            intensity = np.asarray(pcd.colors)[:, 0]
        else:
            logger.debug("No color information, using zero intensity")
            intensity = np.zeros(len(points))
            
        return np.column_stack([points, intensity])
        
    def preprocess(self, points: np.ndarray) -> np.ndarray:
        """
        Preprocess point cloud data.
        
        Args:
            points: (N, 4) array of point coordinates and intensities
            
        Returns:
            Preprocessed point cloud
        """
        start_time = time.time()
        initial_points = len(points)
        logger.info(f"Starting preprocessing of {initial_points} points")
        
        # Remove invalid points
        mask = ~np.isnan(points).any(axis=1)
        points = points[mask]
        removed_nan = initial_points - len(points)
        if removed_nan > 0:
            logger.info(f"Removed {removed_nan} points with NaN values")
        
        # Remove points with zero intensity
        mask = points[:, 3] > 0
        points = points[mask]
        removed_zero = initial_points - removed_nan - len(points)
        if removed_zero > 0:
            logger.info(f"Removed {removed_zero} points with zero intensity")
        
        # Downsample if too many points
        if len(points) > self.max_points:
            logger.info(f"Downsampling from {len(points)} to {self.max_points} points")
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]
            
        # Voxelize point cloud
        logger.info("Starting voxelization")
        points = self._voxelize(points)
        
        preprocess_time = time.time() - start_time
        logger.info(f"Preprocessing completed in {preprocess_time:.2f} seconds")
        logger.info(f"Final point cloud statistics:")
        logger.info(f"  - Points: {len(points)} (reduced by {initial_points - len(points)} points)")
        logger.info(f"  - Coordinate range: X[{points[:,0].min():.2f}, {points[:,0].max():.2f}], "
                   f"Y[{points[:,1].min():.2f}, {points[:,1].max():.2f}], "
                   f"Z[{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
        logger.info(f"  - Intensity range: [{points[:,3].min():.2f}, {points[:,3].max():.2f}]")
        
        return points
        
    def _voxelize(self, points: np.ndarray) -> np.ndarray:
        """
        Voxelize point cloud to reduce density.
        
        Args:
            points: (N, 4) array of point coordinates and intensities
            
        Returns:
            Voxelized point cloud
        """
        start_time = time.time()
        initial_points = len(points)
        logger.info(f"Starting voxelization of {initial_points} points")
        
        # Input validation
        if points.size == 0:
            logger.warning("Empty point cloud received")
            return np.zeros((0, 4))
            
        if np.isnan(points).any() or np.isinf(points).any():
            logger.warning("Point cloud contains NaN or infinite values, removing them")
            mask = ~(np.isnan(points).any(axis=1) | np.isinf(points).any(axis=1))
            points = points[mask]
            logger.info(f"Removed {(initial_points - len(points))} invalid points")
            
        if points.size == 0:
            logger.warning("No valid points after removing NaN/inf values")
            return np.zeros((0, 4))
            
        # Create voxel grid with safe division
        voxel_size = np.array(self.voxel_size)
        logger.info(f"Using voxel size: {voxel_size}")
        
        # Ensure voxel_size is not zero to avoid division by zero
        voxel_size = np.maximum(voxel_size, np.finfo(float).eps)
        
        # Safe division with clipping to avoid overflow
        coords = points[:, :3]
        max_coord = np.finfo(np.float32).max
        coords = np.clip(coords, -max_coord, max_coord)
        voxel_grid = np.floor(coords / voxel_size).astype(int)
        
        # Get unique voxels
        unique_voxels, inverse_indices = np.unique(voxel_grid, axis=0, return_inverse=True)
        logger.info(f"Created {len(unique_voxels)} unique voxels")
        
        # Average points in each voxel with robust mean calculation
        voxelized_points = np.zeros((len(unique_voxels), 4))
        skipped_voxels = 0
        
        for i in range(len(unique_voxels)):
            mask = inverse_indices == i
            voxel_points = points[mask]
            
            # Robust mean calculation for coordinates
            valid_coords = voxel_points[:, :3]
            valid_coords = valid_coords[~np.isnan(valid_coords).any(axis=1)]
            if len(valid_coords) > 0:
                voxelized_points[i, :3] = np.mean(valid_coords, axis=0)
            else:
                logger.debug(f"No valid coordinates in voxel {i}")
                skipped_voxels += 1
                continue
                
            # Use max intensity for the voxel
            valid_intensities = voxel_points[:, 3]
            valid_intensities = valid_intensities[~np.isnan(valid_intensities)]
            if len(valid_intensities) > 0:
                voxelized_points[i, 3] = np.max(valid_intensities)
            else:
                logger.debug(f"No valid intensities in voxel {i}")
                skipped_voxels += 1
                continue
                
        # Remove any voxels that still have NaN values
        valid_mask = ~np.isnan(voxelized_points).any(axis=1)
        voxelized_points = voxelized_points[valid_mask]
        
        voxel_time = time.time() - start_time
        logger.info(f"Voxelization completed in {voxel_time:.2f} seconds")
        logger.info(f"Voxelization statistics:")
        logger.info(f"  - Input points: {initial_points}")
        logger.info(f"  - Unique voxels: {len(unique_voxels)}")
        logger.info(f"  - Skipped voxels: {skipped_voxels}")
        logger.info(f"  - Final points: {len(voxelized_points)}")
        logger.info(f"  - Reduction ratio: {len(voxelized_points)/initial_points:.2%}")
        
        return voxelized_points
        
    def transform_coordinates(self, points: np.ndarray, 
                            source_system: str = 'sensor',
                            target_system: str = 'world') -> np.ndarray:
        """
        Transform point cloud coordinates between different coordinate systems.
        
        Args:
            points: (N, 4) array of point coordinates and intensities
            source_system: Source coordinate system ('sensor' or 'world')
            target_system: Target coordinate system ('sensor' or 'world')
            
        Returns:
            Transformed point cloud
        """
        start_time = time.time()
        logger.info(f"Transforming coordinates from {source_system} to {target_system}")
        
        if source_system == target_system:
            logger.info("Source and target systems are the same, no transformation needed")
            return points
            
        # TODO: Implement coordinate transformations
        # This will be expanded based on the specific requirements
        transform_time = time.time() - start_time
        logger.info(f"Coordinate transformation completed in {transform_time:.2f} seconds")
        return points 