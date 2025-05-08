# LiDAR Processing Pipeline

A robust LiDAR data processing pipeline for crowd management and analysis.

## Features

- Support for multiple LiDAR data formats (.bin, .pcd)
- Advanced point cloud preprocessing
- Voxelization for point cloud downsampling
- Coordinate system transformations
- Comprehensive logging and validation
- Modular and extensible architecture
- User-friendly command-line interface

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

### Command-Line Interface

The easiest way to process your LiDAR data is using the command-line interface:

```bash
# Process a single file
python -m crowd_management.cli path/to/your/lidar.bin

# Process all files in a directory
python -m crowd_management.cli path/to/your/lidar/directory

# Specify output directory
python -m crowd_management.cli path/to/your/lidar.bin --output-dir results

# Use custom configuration
python -m crowd_management.cli path/to/your/lidar.bin --config my_config.json

# Disable visualizations
python -m crowd_management.cli path/to/your/lidar.bin --no-visualizations
```

### Configuration

You can customize the processing pipeline by creating a configuration file. Copy `crowd_management/config_template.json` and modify it according to your needs:

```json
{
    "point_feature_size": 4,
    "max_points": 100000,
    "voxel_size": [0.1, 0.1, 0.1],
    "output_dir": "output",
    "save_visualizations": true,
    "visualization": {
        "point_size": 2,
        "color_map": "viridis",
        "save_format": "png",
        "resolution": [1920, 1080]
    },
    "processing": {
        "remove_nan": true,
        "remove_zero_intensity": true,
        "downsample": true,
        "voxelize": true
    },
    "coordinate_system": {
        "source": "sensor",
        "target": "world"
    }
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

### Output

The processed data is saved in the following format:
- Processed point clouds: `.npy` files
- Visualizations: `.png` files (if enabled)
- Log files: `.log` files with processing details

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
│   ├── cli.py
│   ├── config.py
│   ├── config_template.json
│   └── lidar_processor.py
├── test_lidar_processor.py
├── requirements.txt
└── README.md
```
