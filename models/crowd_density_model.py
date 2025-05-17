import numpy as np
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
from utils.data_processing import extract_people_positions, calculate_grid_density

class CrowdDensityModel:
    """
    Model for analyzing crowd density from LiDAR point cloud data.
    
    This model identifies potential people in the scene and calculates
    crowd density metrics.
    """
    
    def __init__(self, grid_size=1.0):
        """
        Initialize the crowd density model.
        
        Args:
            grid_size (float): Size of grid cells for density calculation in meters
        """
        self.grid_size = grid_size
    
    def analyze(self, processed_data):
        """
        Analyze the processed point cloud data to estimate crowd density.
        
        Args:
            processed_data (dict): Processed point cloud data
            
        Returns:
            dict: Crowd density analysis results
        """
        # Extract people positions from clustered point cloud
        people_positions = extract_people_positions(processed_data)
        
        # If no people detected, return empty results
        if len(people_positions) == 0:
            return {
                'total_people': 0,
                'avg_density': 0.0,
                'max_density': 0.0,
                'density_map': np.zeros((1, 1)),
                'grid_coordinates': (np.array([0]), np.array([0])),
                'density_values': np.array([0]),
                'hotspots': []
            }
        
        # Calculate density on a grid
        x_range = processed_data['dimensions']['x_range']
        y_range = processed_data['dimensions']['y_range']
        
        grid_x, grid_y, density_grid = calculate_grid_density(
            people_positions, x_range, y_range, self.grid_size
        )
        
        # Flatten for easier processing
        flat_density = density_grid.flatten()
        flat_x = np.repeat(grid_x, len(grid_y))
        flat_y = np.tile(grid_y, len(grid_x))
        
        # Calculate density statistics
        total_people = len(people_positions)
        max_density = np.max(flat_density)
        avg_density = np.mean(flat_density[flat_density > 0]) if np.any(flat_density > 0) else 0
        
        # Identify hotspots (areas with high density)
        hotspot_threshold = max(0.5, avg_density * 1.5)  # At least 50% above average
        hotspot_indices = np.where(flat_density >= hotspot_threshold)[0]
        
        hotspots = []
        for idx in hotspot_indices:
            hotspots.append({
                'x': flat_x[idx],
                'y': flat_y[idx],
                'density': flat_density[idx]
            })
        
        # Sort hotspots by density (highest first)
        hotspots = sorted(hotspots, key=lambda x: x['density'], reverse=True)
        
        # Limit to top 5 hotspots
        hotspots = hotspots[:5]
        
        # Create density values array for combined metrics
        density_values = flat_density
        grid_coordinates = (flat_x, flat_y)
        
        results = {
            'total_people': total_people,
            'avg_density': avg_density,
            'max_density': max_density,
            'density_map': density_grid,
            'grid_coordinates': grid_coordinates,
            'density_values': density_values,
            'hotspots': hotspots
        }
        
        return results
    
    def calculate_risk_level(self, density):
        """
        Calculate risk level based on crowd density.
        
        Args:
            density (float): Crowd density in people per square meter
            
        Returns:
            str: Risk level ('Low', 'Moderate', 'High', 'Critical')
        """
        if density < 1.0:
            return 'Low'
        elif density < 2.5:
            return 'Moderate'
        elif density < 4.0:
            return 'High'
        else:
            return 'Critical'
