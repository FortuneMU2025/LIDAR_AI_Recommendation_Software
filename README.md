# LiDAR Processing Pipeline

A robust LiDAR data processing pipeline for crowd management and analysis.

## Features

- Support for multiple LiDAR data formats (.bin, .pcd)
- Advanced point cloud preprocessing
- Voxelization for point cloud downsampling
- Coordinate system transformations
- Comprehensive logging and validation
- Modular and extensible architecture

## Installation

1. Clone the repository:
```bash
git clone https://github.com/FortuneMU2025/LIDAR_AI_Recommendation_Software.git
cd LIDAR_AI_Recommendation_Software
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Unix/MacOS:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from crowd_management.lidar_processor import LiDARProcessor
from crowd_management.config import DEFAULT_CONFIG

# Initialize processor
processor = LiDARProcessor(DEFAULT_CONFIG)

# Load and process LiDAR data
points = processor.load_data("path/to/your/lidar.bin")
processed_points = processor.preprocess(points)
transformed_points = processor.transform_coordinates(processed_points)
```

### Configuration

The pipeline can be configured through the `DEFAULT_CONFIG` dictionary:

```python
config = {
    'point_feature_size': 4,  # x, y, z, intensity
    'max_points': 100000,     # Maximum points to process
    'voxel_size': [0.1, 0.1, 0.1],  # Voxel size for downsampling
    'output_dir': 'output',   # Output directory
    'save_visualizations': True  # Save visualization results
}
```

### Data Formats

The pipeline supports two main LiDAR data formats:

1. Binary (.bin):
   - Raw point cloud data
   - Each point has 4 features (x, y, z, intensity)
   - Common in autonomous vehicle datasets

2. PCD (.pcd):
   - Point Cloud Data format
   - Supports additional features like color
   - Standard format for point cloud processing

### Processing Pipeline

1. **Data Loading**:
   - Validates file format and existence
   - Loads points with coordinates and intensity
   - Performs initial data validation

2. **Preprocessing**:
   - Removes invalid points (NaN values)
   - Filters points with zero intensity
   - Downsamples if point count exceeds limit
   - Voxelizes point cloud for density reduction

3. **Coordinate Transformation**:
   - Transforms between different coordinate systems
   - Supports sensor-to-world transformations
   - Handles coordinate system conversions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Project Structure

```
.
├── crowd_management/
│   ├── __init__.py
│   ├── config.py
│   └── lidar_processor.py
├── test_lidar_processor.py
├── requirements.txt
└── README.md
```
