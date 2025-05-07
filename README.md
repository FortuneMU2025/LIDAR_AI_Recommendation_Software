# LiDAR Processing Module

A Python module for processing and analyzing LiDAR point cloud data.

## Features

- Load LiDAR data from .bin and .pcd formats
- Preprocess point clouds (filtering, downsampling, voxelization)
- Basic coordinate transformations
- Point cloud visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lidar_processing.git
cd lidar_processing
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
.\venv\Scripts\activate   # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Development Setup

1. Install development dependencies:
```bash
pip install -r requirements.txt
```

2. Run tests:
```bash
pytest test_lidar_processor.py
```

3. Format code:
```bash
black .
flake8
```

## Usage

Basic usage example:
```python
from crowd_management.lidar_processor import LiDARProcessor
from crowd_management.config import DEFAULT_CONFIG

# Initialize processor
processor = LiDARProcessor(DEFAULT_CONFIG)

# Load and process point cloud
points = processor.load_data("path/to/pointcloud.bin")
processed_points = processor.preprocess(points)
```

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
