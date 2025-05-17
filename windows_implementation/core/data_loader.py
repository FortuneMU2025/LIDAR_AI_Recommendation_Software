"""
Data Loader module for LiDAR Crowd Analytics
Handles loading and parsing various LiDAR data formats
"""

import os
import numpy as np
import pandas as pd
import logging
import struct
import re

logger = logging.getLogger(__name__)

class Dataset:
    """Represents a dataset with point cloud data"""
    
    def __init__(self, points, metadata=None):
        """
        Initialize a dataset
        
        Args:
            points (numpy.ndarray): Point cloud data as (n, 3) array
            metadata (dict, optional): Additional metadata about the dataset
        """
        self.points = points
        self.metadata = metadata or {}


class DataLoader:
    """Loads and parses various LiDAR data formats"""
    
    def __init__(self):
        """Initialize data loader"""
        pass
    
    def load_file(self, file_path):
        """
        Load a point cloud file
        
        Args:
            file_path (str): Path to the point cloud file
            
        Returns:
            Dataset: Dataset object with the loaded point cloud
            
        Raises:
            ValueError: If the file format is not supported
            FileNotFoundError: If the file does not exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file format from extension
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.csv':
            return self._load_csv(file_path)
        elif ext == '.xyz' or ext == '.txt':
            return self._load_xyz(file_path)
        elif ext == '.pcd':
            return self._load_pcd(file_path)
        elif ext == '.ply':
            return self._load_ply(file_path)
        elif ext == '.las' or ext == '.laz':
            return self._load_las(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def _load_csv(self, file_path):
        """
        Load a CSV file
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            Dataset: Dataset object with the loaded point cloud
        """
        try:
            logger.info(f"Loading CSV file: {file_path}")
            
            # Try to determine column names
            df = pd.read_csv(file_path, nrows=0)
            headers = df.columns.tolist()
            
            # Look for x, y, z columns
            x_col, y_col, z_col = None, None, None
            for header in headers:
                header_lower = header.lower()
                if header_lower == 'x':
                    x_col = header
                elif header_lower == 'y':
                    y_col = header
                elif header_lower == 'z':
                    z_col = header
            
            # If we found x, y, z columns, use them
            if x_col and y_col and z_col:
                df = pd.read_csv(file_path, usecols=[x_col, y_col, z_col])
                points = df[[x_col, y_col, z_col]].values
            else:
                # Try first three columns
                df = pd.read_csv(file_path)
                if len(df.columns) >= 3:
                    points = df.iloc[:, :3].values
                else:
                    raise ValueError("CSV file doesn't have at least 3 columns for X, Y, Z coordinates")
            
            # Create metadata
            metadata = {
                'file_format': 'csv',
                'file_path': file_path,
                'point_count': len(points),
                'columns': headers
            }
            
            # Create and return dataset
            return Dataset(points, metadata)
            
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise
    
    def _load_xyz(self, file_path):
        """
        Load an XYZ/TXT file
        
        Args:
            file_path (str): Path to the XYZ/TXT file
            
        Returns:
            Dataset: Dataset object with the loaded point cloud
        """
        try:
            logger.info(f"Loading XYZ/TXT file: {file_path}")
            
            # Determine the delimiter by reading the first line
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                if ',' in first_line:
                    delimiter = ','
                elif ';' in first_line:
                    delimiter = ';'
                else:
                    delimiter = None  # Space or tab
            
            # Load points using numpy
            points = np.loadtxt(file_path, delimiter=delimiter)
            
            # If more than 3 columns, take only the first 3 (X, Y, Z)
            if points.shape[1] > 3:
                points = points[:, :3]
            
            # Create metadata
            metadata = {
                'file_format': 'xyz',
                'file_path': file_path,
                'point_count': len(points),
                'delimiter': delimiter
            }
            
            # Create and return dataset
            return Dataset(points, metadata)
            
        except Exception as e:
            logger.error(f"Error loading XYZ/TXT file: {str(e)}")
            raise
    
    def _load_pcd(self, file_path):
        """
        Load a PCD file
        
        Args:
            file_path (str): Path to the PCD file
            
        Returns:
            Dataset: Dataset object with the loaded point cloud
        """
        try:
            logger.info(f"Loading PCD file: {file_path}")
            
            # Parse PCD header
            header = {}
            points = []
            data_section_started = False
            
            with open(file_path, 'rb') as f:
                for line in f:
                    # Decode line if it's a bytes object
                    if isinstance(line, bytes):
                        line = line.decode('utf-8', errors='ignore')
                    
                    # Check if we've reached the data section
                    if data_section_started:
                        # Parse data line
                        values = line.strip().split()
                        if len(values) >= 3:
                            # Try to convert values to float
                            try:
                                point = [float(values[0]), float(values[1]), float(values[2])]
                                points.append(point)
                            except ValueError:
                                continue
                        continue
                    
                    # Parse header
                    if line.startswith('#'):
                        continue
                    
                    if line.strip() == 'DATA ascii':
                        data_section_started = True
                        continue
                    
                    if line.strip() == 'DATA binary':
                        # Binary PCD format - more complex parsing needed
                        # This is a simplified implementation that only handles ASCII formats
                        raise ValueError("Binary PCD format not supported by this implementation")
                    
                    # Parse header field
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        header[parts[0].lower()] = ' '.join(parts[1:])
            
            # Convert points to numpy array
            if not points:
                raise ValueError("No valid points found in PCD file")
            
            points_array = np.array(points, dtype=float)
            
            # Create metadata
            metadata = {
                'file_format': 'pcd',
                'file_path': file_path,
                'point_count': len(points_array),
                'header': header
            }
            
            # Create and return dataset
            return Dataset(points_array, metadata)
            
        except Exception as e:
            logger.error(f"Error loading PCD file: {str(e)}")
            raise
    
    def _load_ply(self, file_path):
        """
        Load a PLY file
        
        Args:
            file_path (str): Path to the PLY file
            
        Returns:
            Dataset: Dataset object with the loaded point cloud
        """
        try:
            logger.info(f"Loading PLY file: {file_path}")
            
            # Parse PLY header
            header = {}
            vertex_count = 0
            x_prop, y_prop, z_prop = None, None, None
            data_format = 'ascii'
            header_end = False
            
            with open(file_path, 'rb') as f:
                for line in f:
                    # Decode line if it's a bytes object
                    if isinstance(line, bytes):
                        line = line.decode('utf-8', errors='ignore')
                    
                    # Check if we've reached the end of the header
                    if header_end:
                        break
                    
                    # Parse header line
                    line = line.strip()
                    
                    if line == 'end_header':
                        header_end = True
                        continue
                    
                    if line.startswith('format'):
                        parts = line.split()
                        if len(parts) >= 2:
                            data_format = parts[1]
                    
                    if line.startswith('element vertex'):
                        parts = line.split()
                        if len(parts) >= 3:
                            vertex_count = int(parts[2])
                    
                    if line.startswith('property float') or line.startswith('property double'):
                        parts = line.split()
                        if len(parts) >= 3:
                            prop_name = parts[2]
                            if prop_name.lower() == 'x':
                                x_prop = True
                            elif prop_name.lower() == 'y':
                                y_prop = True
                            elif prop_name.lower() == 'z':
                                z_prop = True
            
            # Check if we have valid X, Y, Z properties
            if not (x_prop and y_prop and z_prop):
                raise ValueError("PLY file doesn't have valid X, Y, Z properties")
            
            # Handle different data formats
            if data_format != 'ascii':
                # This is a simplified implementation that only handles ASCII formats
                raise ValueError(f"PLY format '{data_format}' not supported by this implementation")
            
            # Read vertex data
            points = []
            with open(file_path, 'rb') as f:
                # Skip to data section
                for line in f:
                    if isinstance(line, bytes):
                        line = line.decode('utf-8', errors='ignore')
                    if line.strip() == 'end_header':
                        break
                
                # Read vertices
                for _ in range(vertex_count):
                    line = f.readline()
                    if isinstance(line, bytes):
                        line = line.decode('utf-8', errors='ignore')
                    
                    values = line.strip().split()
                    if len(values) >= 3:
                        try:
                            point = [float(values[0]), float(values[1]), float(values[2])]
                            points.append(point)
                        except ValueError:
                            continue
            
            # Convert points to numpy array
            if not points:
                raise ValueError("No valid points found in PLY file")
            
            points_array = np.array(points, dtype=float)
            
            # Create metadata
            metadata = {
                'file_format': 'ply',
                'file_path': file_path,
                'point_count': len(points_array),
                'vertex_count': vertex_count,
                'data_format': data_format
            }
            
            # Create and return dataset
            return Dataset(points_array, metadata)
            
        except Exception as e:
            logger.error(f"Error loading PLY file: {str(e)}")
            raise
    
    def _load_las(self, file_path):
        """
        Load a LAS/LAZ file
        
        Args:
            file_path (str): Path to the LAS/LAZ file
            
        Returns:
            Dataset: Dataset object with the loaded point cloud
            
        Note:
            This is a simplified implementation. For full LAS/LAZ support,
            the laspy library should be used.
        """
        try:
            logger.info(f"Loading LAS/LAZ file: {file_path}")
            
            # Check if this is a LAZ file
            if file_path.lower().endswith('.laz'):
                raise ValueError("LAZ files require the laspy library with laszip support")
            
            # Note: This is a simple implementation for demonstration purposes
            # A full implementation would use the laspy library to properly parse LAS/LAZ files
            
            # For now, we'll read just enough of the header to determine point data format
            with open(file_path, 'rb') as f:
                # Check file signature
                file_signature = f.read(4).decode()
                if file_signature != 'LASF':
                    raise ValueError("Invalid LAS file signature")
                
                # Skip to point data format ID (byte 104)
                f.seek(104)
                point_data_format_id = struct.unpack('<B', f.read(1))[0]
                
                # Read point data record length
                point_data_record_length = struct.unpack('<H', f.read(2))[0]
                
                # Read number of point records
                f.seek(107)
                num_point_records = struct.unpack('<I', f.read(4))[0]
                
                # Skip to point data offset (byte 96)
                f.seek(96)
                point_data_offset = struct.unpack('<I', f.read(4))[0]
                
                # Skip to point data
                f.seek(point_data_offset)
                
                # Read point data
                points = []
                for _ in range(min(num_point_records, 10000)):  # Limit to 10000 points for demonstration
                    # Read X, Y, Z coordinates (scaled integers)
                    point_data = f.read(point_data_record_length)
                    if len(point_data) < 12:  # At least need X, Y, Z (4 bytes each)
                        break
                    
                    x = struct.unpack('<i', point_data[0:4])[0]
                    y = struct.unpack('<i', point_data[4:8])[0]
                    z = struct.unpack('<i', point_data[8:12])[0]
                    
                    # Apply default scale factor of 0.01 for demonstration
                    # (In a proper implementation, we would read the actual scale factors from the header)
                    scale_factor = 0.01
                    points.append([x * scale_factor, y * scale_factor, z * scale_factor])
            
            # Convert points to numpy array
            if not points:
                raise ValueError("No valid points found in LAS file")
            
            points_array = np.array(points, dtype=float)
            
            # Create metadata
            metadata = {
                'file_format': 'las',
                'file_path': file_path,
                'point_count': len(points_array),
                'point_data_format_id': point_data_format_id,
                'total_points': num_point_records
            }
            
            # Create and return dataset
            return Dataset(points_array, metadata)
            
        except Exception as e:
            logger.error(f"Error loading LAS/LAZ file: {str(e)}")
            if "LAZ files require the laspy library" in str(e):
                raise ValueError("LAZ files require additional libraries. Please install with: pip install laspy[laszip]")
            raise