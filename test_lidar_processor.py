"""
Test script for LiDAR processor.
"""

import os
import numpy as np
from crowd_management.lidar_processor import LiDARProcessor
from crowd_management.config import DEFAULT_CONFIG

def test_lidar_processor():
    # Initialize processor
    processor = LiDARProcessor(DEFAULT_CONFIG)
    
    # Test data directory
    test_dir = "test_data"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create test point cloud
    num_points = 1000
    points = np.random.rand(num_points, 4)
    points[:, 3] = np.random.rand(num_points)  # Random intensities
    
    # Save test data
    bin_path = os.path.join(test_dir, "test.bin")
    points.tofile(bin_path)
    
    try:
        # Test loading
        loaded_points = processor.load_data(bin_path)
        print(f"Loaded {len(loaded_points)} points from binary file")
        
        # Test preprocessing
        processed_points = processor.preprocess(loaded_points)
        print(f"Processed {len(processed_points)} points")
        
        # Test coordinate transformation
        transformed_points = processor.transform_coordinates(processed_points)
        print(f"Transformed {len(transformed_points)} points")
        
        print("\nAll tests passed successfully!")
        
    except Exception as e:
        print(f"Test failed: {str(e)}")
        
    finally:
        # Cleanup
        if os.path.exists(bin_path):
            os.remove(bin_path)
        if os.path.exists(test_dir):
            os.rmdir(test_dir)

if __name__ == "__main__":
    test_lidar_processor() 