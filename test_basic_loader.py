"""
Basic test script for LiDAR data loading.
"""

import os
import numpy as np
from pathlib import Path
from crowd_management.data.loaders import LoaderFactory

def test_basic_loader():
    # Initialize loader factory
    print("Initializing loader factory...")
    
    # Create test data directory
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create test point cloud with valid ranges
    num_points = 1000
    # Generate points within valid ranges (-100 to 100 for coordinates, 0 to 1 for intensity)
    coords = np.random.uniform(-100, 100, (num_points, 3))
    intensity = np.random.uniform(0, 1, (num_points, 1))
    points = np.hstack([coords, intensity])
    
    # Save test data
    bin_path = test_dir / "test.bin"
    points.astype(np.float32).tofile(bin_path)
    
    try:
        # Create loader
        print(f"\nCreating loader for {bin_path}...")
        loader = LoaderFactory.create_loader(bin_path)
        print(f"Created loader: {type(loader).__name__}")
        
        # Load data
        print("\nLoading point cloud data...")
        loaded_points = loader.load(bin_path)
        print(f"Successfully loaded {len(loaded_points)} points")
        print(f"Point cloud shape: {loaded_points.shape}")
        
        # Get metadata
        print("\nGetting point cloud metadata...")
        metadata = loader.get_metadata(bin_path)
        print("Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        print("\nAll tests passed successfully!")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        
    finally:
        # Cleanup
        if bin_path.exists():
            bin_path.unlink()
        if test_dir.exists():
            test_dir.rmdir()

if __name__ == "__main__":
    test_basic_loader() 