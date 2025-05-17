# LiDAR Crowd Analytics - Windows Application

## Overview

LiDAR Crowd Analytics is a professional Windows desktop application for analyzing 3D LiDAR datasets to generate crowd analytics and management strategies for event managers. This application helps event planners understand crowd density, movement patterns, and identify potential bottlenecks or safety concerns.

## Key Features

- **Advanced Data Import**: Support for industry-standard LiDAR formats (LAS, LAZ, PCD, PLY, XYZ, CSV)
- **3D Visualization**: Interactive 3D point cloud visualization with customizable views
- **Crowd Density Analysis**: Identify crowd density hotspots and calculate occupancy metrics
- **Flow Pattern Analysis**: Analyze movement vectors and identify potential bottlenecks
- **Comprehensive Reporting**: Generate detailed PDF and HTML reports with actionable recommendations
- **Project Management**: Save, load, and manage multiple analysis projects
- **Database Integration**: Store analysis results in a local database for future reference

## System Requirements

- Windows 10/11 (64-bit)
- 8GB RAM (16GB recommended)
- OpenGL 4.0 compatible graphics card
- 4GB free disk space
- Intel i5/AMD Ryzen 5 or equivalent processor

## Installation

1. **Install Python**: 
   - Download and install Python 3.11 or newer from [python.org](https://www.python.org/downloads/)
   - Make sure to check "Add Python to PATH" during installation

2. **Install Required Libraries**:
   ```
   pip install numpy pandas scipy scikit-learn matplotlib plotly PyQt5
   ```

3. **Additional Libraries for Advanced Features**:
   ```
   pip install laspy[laszip] open3d
   ```

4. **Download and Extract Application**:
   - Extract the application files to your preferred location

## Getting Started

1. **Launch the Application**:
   - Navigate to the application directory
   - Run `python main.py` to start the application

2. **Create a New Project**:
   - Click "File" > "New Project" to create a new analysis project
   - Enter a project name and select a location to save it

3. **Import Data**:
   - Click "File" > "Import Data" or use the Import button in the ribbon
   - Select your LiDAR dataset file (.las, .pcd, .ply, .xyz, .csv)

4. **Visualize the Data**:
   - Use the 3D visualization window to explore your point cloud
   - Change visualization settings using the options panel

5. **Run Analysis**:
   - Go to the "Analysis" tab
   - Configure analysis parameters
   - Click "Run Analysis" to process the data

6. **View Results**:
   - Examine crowd density heatmaps and flow patterns
   - Review identified hotspots and bottlenecks

7. **Generate Reports**:
   - Click "Generate Report" to create a comprehensive analysis report
   - Choose between PDF and HTML formats
   - Save and share the report with stakeholders

## File Format Support

The application supports the following LiDAR data formats:

- **CSV**: Comma-separated values with X, Y, Z columns
- **XYZ/TXT**: Simple text files with X, Y, Z coordinates
- **PCD**: Point Cloud Data format (ASCII)
- **PLY**: Polygon File Format (ASCII)
- **LAS/LAZ**: Industry-standard LiDAR format (requires laspy library)

## Technical Support

For technical assistance or to report bugs, please contact:
- Email: support@example.com
- Website: https://example.com/support

## License

This software is proprietary and confidential. Unauthorized copying, distribution, or use is strictly prohibited.

Copyright © 2023-2025 Your Organization
All rights reserved.