import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial import Delaunay
from utils.data_processing import extract_people_positions

class CrowdFlowModel:
    """
    Model for analyzing crowd flow patterns from LiDAR point cloud data.
    
    This model estimates movement vectors and identifies potential bottlenecks.
    """
    
    def __init__(self):
        """Initialize the crowd flow model."""
        # Previous state for tracking
        self.prev_positions = None
        self.flow_vectors = None
        
        # Simulated flow field parameters (for demonstration)
        # In a real implementation, this would be derived from sequential point clouds
        self.simulation_params = {
            'flow_field_complexity': 2,
            'bottleneck_count': 3,
            'flow_speed_range': (0.2, 1.5),  # meters per second
            'random_seed': 42
        }
    
    def analyze(self, processed_data):
        """
        Analyze the processed point cloud data to estimate crowd flow.
        
        Args:
            processed_data (dict): Processed point cloud data
            
        Returns:
            dict: Crowd flow analysis results
        """
        # Extract people positions from clustered point cloud
        people_positions = extract_people_positions(processed_data)
        
        # If no people detected, return empty results
        if len(people_positions) == 0:
            return {
                'flow_vectors': {
                    'positions': np.zeros((0, 2)),
                    'vectors': np.zeros((0, 2)),
                    'magnitudes': np.zeros(0)
                },
                'avg_speed': 0.0,
                'dominant_direction': 'N/A',
                'bottlenecks': []
            }
        
        # Generate simulated flow vectors
        # In a real implementation, this would use temporal tracking between frames
        flow_vectors = self._generate_simulated_flow(people_positions, processed_data)
        
        # Calculate flow statistics
        magnitudes = flow_vectors['magnitudes']
        vectors = flow_vectors['vectors']
        
        avg_speed = np.mean(magnitudes)
        
        # Calculate dominant direction
        if len(vectors) > 0:
            avg_vector = np.mean(vectors, axis=0)
            angle = np.arctan2(avg_vector[1], avg_vector[0]) * 180 / np.pi
            
            # Convert angle to cardinal direction
            directions = ["E", "NE", "N", "NW", "W", "SW", "S", "SE", "E"]
            idx = int((angle + 22.5) % 360 / 45)
            dominant_direction = directions[idx]
        else:
            dominant_direction = "N/A"
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(flow_vectors, processed_data)
        
        results = {
            'flow_vectors': flow_vectors,
            'avg_speed': avg_speed,
            'dominant_direction': dominant_direction,
            'bottlenecks': bottlenecks
        }
        
        return results
    
    def _generate_simulated_flow(self, people_positions, processed_data):
        """
        Generate simulated flow vectors for demonstration.
        
        Args:
            people_positions (numpy.ndarray): Array of (x, y) positions of people
            processed_data (dict): Processed point cloud data
            
        Returns:
            dict: Flow vector data
        """
        # Set random seed for reproducibility
        np.random.seed(self.simulation_params['random_seed'])
        
        # Get dimensions
        x_range = processed_data['dimensions']['x_range']
        y_range = processed_data['dimensions']['y_range']
        
        # Create grid points for flow field
        grid_size = 1.0  # meters
        x_grid = np.arange(x_range[0], x_range[1] + grid_size, grid_size)
        y_grid = np.arange(y_range[0], y_range[1] + grid_size, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        grid_points = np.vstack([X.ravel(), Y.ravel()]).T
        
        # Define flow field using sine waves for demonstration
        # In a real implementation, this would be derived from temporal data
        complexity = self.simulation_params['flow_field_complexity']
        
        # Base flow field (everyone moving towards exit)
        # Assume exit is at the center of right edge
        exit_x = x_range[1]
        exit_y = (y_range[0] + y_range[1]) / 2
        
        # Calculate direction vectors towards exit
        vectors = np.zeros((len(grid_points), 2))
        
        for i, (x, y) in enumerate(grid_points):
            # Base vector pointing towards exit
            dx = exit_x - x
            dy = exit_y - y
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance > 0:
                # Normalize
                dx /= distance
                dy /= distance
                
                # Add some variations using sine waves
                angle_mod = np.sin(x * complexity) * np.cos(y * complexity) * 0.5
                
                # Rotate vector by angle_mod
                cos_mod = np.cos(angle_mod)
                sin_mod = np.sin(angle_mod)
                
                dx_new = dx * cos_mod - dy * sin_mod
                dy_new = dx * sin_mod + dy * cos_mod
                
                vectors[i] = [dx_new, dy_new]
            else:
                vectors[i] = [0, 0]
        
        # Add bottlenecks
        for _ in range(self.simulation_params['bottleneck_count']):
            # Random bottleneck position
            bottleneck_x = np.random.uniform(x_range[0] + 1, x_range[1] - 1)
            bottleneck_y = np.random.uniform(y_range[0] + 1, y_range[1] - 1)
            
            # Reduce speed near bottleneck
            for i, (x, y) in enumerate(grid_points):
                # Distance to bottleneck
                dist = np.sqrt((x - bottleneck_x)**2 + (y - bottleneck_y)**2)
                
                # If within bottleneck radius
                if dist < 3.0:
                    # Reduce speed based on distance
                    speed_factor = dist / 3.0
                    vectors[i] *= speed_factor
        
        # Calculate vector magnitudes (speed)
        magnitudes = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)
        
        # Scale to realistic walking speeds
        speed_min, speed_max = self.simulation_params['flow_speed_range']
        scale_factor = (speed_max - speed_min) / np.max(magnitudes) if np.max(magnitudes) > 0 else 1.0
        vectors *= scale_factor
        magnitudes = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)
        magnitudes = np.clip(magnitudes, speed_min, speed_max)
        
        # Create result
        flow_vectors = {
            'positions': grid_points,
            'vectors': vectors,
            'magnitudes': magnitudes
        }
        
        return flow_vectors
    
    def _identify_bottlenecks(self, flow_vectors, processed_data):
        """
        Identify potential bottlenecks in crowd flow.
        
        Args:
            flow_vectors (dict): Flow vector data
            processed_data (dict): Processed point cloud data
            
        Returns:
            list: List of identified bottlenecks
        """
        positions = flow_vectors['positions']
        magnitudes = flow_vectors['magnitudes']
        vectors = flow_vectors['vectors']
        
        # Find areas with low speed but high speed gradient
        # (indicating rapid slowing down)
        
        # Create KD-tree for nearest neighbor search
        kdtree = KDTree(positions)
        
        bottlenecks = []
        
        # For each point, check if it's a potential bottleneck
        for i, (pos, mag) in enumerate(zip(positions, magnitudes)):
            # Skip points with high speed
            if mag > 0.5:  # More than 0.5 m/s is not a bottleneck
                continue
            
            # Get neighbors within 3 meters
            indices = kdtree.query_radius([pos], r=3.0)[0]
            
            # Skip if not enough neighbors
            if len(indices) < 5:
                continue
            
            # Calculate average speed of neighbors
            neighbor_speeds = magnitudes[indices]
            avg_neighbor_speed = np.mean(neighbor_speeds)
            
            # Calculate speed gradient (how much speed reduces towards this point)
            # Find neighbors that are further away
            far_indices = kdtree.query_radius([pos], r=5.0)[0]
            far_indices = np.setdiff1d(far_indices, indices)  # Exclude close neighbors
            
            if len(far_indices) < 3:
                continue
            
            far_neighbor_speeds = magnitudes[far_indices]
            avg_far_speed = np.mean(far_neighbor_speeds)
            
            # Speed gradient
            speed_gradient = avg_far_speed - avg_neighbor_speed
            
            # Calculate flow convergence (vectors pointing towards a common area)
            neighbor_vectors = vectors[indices]
            neighbor_positions = positions[indices]
            
            convergence = 0
            
            for j, neighbor_pos in enumerate(neighbor_positions):
                # Direction from neighbor to current point
                direction = pos - neighbor_pos
                direction_norm = np.linalg.norm(direction)
                
                if direction_norm > 0:
                    direction /= direction_norm
                    
                    # Dot product with neighbor's flow vector
                    # Higher dot product means the vector is pointing towards this point
                    dot_product = np.dot(direction, neighbor_vectors[j])
                    convergence += max(0, dot_product)  # Only count positive convergence
            
            # Normalize convergence by neighbor count
            convergence /= len(indices)
            
            # Combine metrics to calculate bottleneck severity
            severity = (speed_gradient * 5 + convergence * 5) / 2
            
            # If severity is significant, mark as bottleneck
            if severity > 1.0:
                bottlenecks.append({
                    'x': pos[0],
                    'y': pos[1],
                    'severity': min(10, round(severity))  # Scale to 1-10
                })
        
        # Sort bottlenecks by severity (highest first)
        bottlenecks = sorted(bottlenecks, key=lambda x: x['severity'], reverse=True)
        
        # Limit to top 5 bottlenecks
        bottlenecks = bottlenecks[:5]
        
        return bottlenecks
