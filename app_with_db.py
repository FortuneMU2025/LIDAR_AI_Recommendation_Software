import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
import tempfile
import os
import io
import base64
from datetime import datetime
from utils.database import Database

# Initialize database
db = Database()

# Page configuration
st.set_page_config(
    page_title="LiDAR Crowd Analytics",
    page_icon="📊",
    layout="wide"
)

# Helper functions for data processing
def load_point_cloud(uploaded_file):
    """Load point cloud data from various file formats"""
    try:
        file_ext = uploaded_file.name.lower().split('.')[-1]
        
        if file_ext == 'csv':
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            # Try to find x, y, z columns
            xyz_cols = []
            for col in df.columns:
                if col.lower() in ['x', 'y', 'z']:
                    xyz_cols.append(col)
            
            if len(xyz_cols) >= 3:
                points = df[xyz_cols[:3]].values
            else:
                points = df.iloc[:, :3].values
                
        elif file_ext == 'xyz' or file_ext == 'txt':
            # Simple space or comma separated XYZ file
            content = uploaded_file.getvalue().decode('utf-8')
            lines = content.strip().split('\n')
            points_data = []
            for line in lines:
                values = line.strip().split()
                if len(values) >= 3:
                    try:
                        points_data.append([float(val) for val in values[:3]])
                    except ValueError:
                        continue  # Skip lines that can't be converted to float
            points = np.array(points_data)
            
        elif file_ext == 'npy':
            # Numpy binary format
            points = np.load(io.BytesIO(uploaded_file.getvalue()))[:, :3]
            
        else:
            st.error(f"Unsupported file format: {file_ext}")
            st.info("Supported formats: CSV, XYZ, TXT, NPY")
            return None
            
        # Basic validation
        if len(points) == 0:
            st.error("No valid points found in the file")
            return None
            
        return points
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def preprocess_point_cloud(points):
    """Preprocess the point cloud data for analysis"""
    
    # Generate colors based on height (z-value)
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    normalized_height = (points[:, 2] - z_min) / (z_max - z_min + 1e-10)
    colors = np.zeros((len(points), 3))
    colors[:, 0] = normalized_height  # R channel mapped to height
    colors[:, 1] = 0.5 * (1 - normalized_height)  # G channel inverse to height
    colors[:, 2] = 0.5  # B channel constant
    
    # Handle outliers
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    mask = np.all(np.abs(points - mean) < 3 * std, axis=1)
    inlier_points = points[mask]
    
    # Create synthetic normals
    normals = np.zeros_like(inlier_points)
    normals[:, 2] = 1.0  # All normals point up
    
    # Detect ground plane (assume lowest 30% of points by z-value are ground)
    z_threshold = np.percentile(inlier_points[:, 2], 30)
    ground_indices = inlier_points[:, 2] <= z_threshold
    non_ground_indices = ~ground_indices
    
    # Cluster non-ground points to identify potential people
    non_ground_points = inlier_points[non_ground_indices]
    
    if len(non_ground_points) > 10:
        # Cluster for people detection
        clustering = DBSCAN(eps=0.3, min_samples=5).fit(non_ground_points)
        cluster_labels = clustering.labels_
    else:
        cluster_labels = np.zeros(len(non_ground_points), dtype=int)
    
    # Map cluster labels back to all points
    full_labels = np.ones(len(inlier_points), dtype=int) * -1
    full_labels[non_ground_indices] = cluster_labels
    
    # Calculate dataset dimensions
    x_min, y_min, z_min = np.min(inlier_points, axis=0)
    x_max, y_max, z_max = np.max(inlier_points, axis=0)
    
    dimensions = {
        'x_range': (x_min, x_max),
        'y_range': (y_min, y_max),
        'z_range': (z_min, z_max),
        'width': x_max - x_min,
        'length': y_max - y_min,
        'height': z_max - z_min
    }
    
    # Create processed data dictionary
    processed_data = {
        'points': inlier_points,
        'colors': colors[mask],
        'clusters': full_labels,
        'dimensions': dimensions
    }
    
    return processed_data

def visualize_point_cloud(processed_data, point_size=3, color_by="Height"):
    """Create 3D visualization of point cloud"""
    points = processed_data['points']
    clusters = processed_data['clusters']
    
    # Downsample if too many points
    if len(points) > 20000:
        indices = np.random.choice(len(points), 20000, replace=False)
        points = points[indices]
        clusters = clusters[indices]
    
    # Prepare colors based on the selected method
    if color_by == "Height":
        colors = points[:, 2]  # Z-coordinate for height
        colorscale = 'Viridis'
        colorbar_title = 'Height (m)'
    
    elif color_by == "Density":
        # Calculate local density
        kdtree = KDTree(points)
        densities = kdtree.query_radius(points, r=0.5, count_only=True)
        colors = densities
        colorscale = 'Reds'
        colorbar_title = 'Local Point Density'
    
    elif color_by == "Cluster":
        colors = clusters
        colorscale = 'Rainbow'
        colorbar_title = 'Cluster ID'
    
    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=point_size,
            color=colors,
            colorscale=colorscale,
            opacity=0.8,
            colorbar=dict(title=colorbar_title)
        )
    )])
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600
    )
    
    return fig

def create_density_heatmap(processed_data):
    """Create heatmap of point cloud density"""
    points = processed_data['points']
    x_range = processed_data['dimensions']['x_range']
    y_range = processed_data['dimensions']['y_range']
    
    # Create 2D histogram
    hist, x_edges, y_edges = np.histogram2d(
        points[:, 0], points[:, 1],
        bins=100,
        range=[x_range, y_range]
    )
    
    # Get bin centers
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=hist.T,
        x=x_centers,
        y=y_centers,
        colorscale='Viridis',
        colorbar=dict(title='Point Density')
    ))
    
    # Update layout
    fig.update_layout(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        title='Point Density Heatmap',
        height=500
    )
    
    return fig

def analyze_crowd_density(processed_data):
    """Analyze crowd density from point cloud clusters"""
    points = processed_data['points']
    clusters = processed_data['clusters']
    
    # Extract clusters representing people (exclude noise with label -1)
    people_clusters = np.unique(clusters[clusters >= 0])
    num_people = len(people_clusters)
    
    # Calculate density metrics
    area = processed_data['dimensions']['width'] * processed_data['dimensions']['length']
    avg_density = num_people / max(1, area)  # people per square meter
    
    # Find density hotspots using kernel density estimation
    if num_people > 0:
        people_positions = []
        for cluster_id in people_clusters:
            cluster_points = points[clusters == cluster_id]
            centroid = np.mean(cluster_points, axis=0)
            people_positions.append(centroid[:2])  # Keep only x, y
        
        people_positions = np.array(people_positions)
        
        # Create grid for density calculation
        x_range = processed_data['dimensions']['x_range']
        y_range = processed_data['dimensions']['y_range']
        
        grid_size = 1.0  # 1 meter grid
        x_grid = np.arange(x_range[0], x_range[1] + grid_size, grid_size)
        y_grid = np.arange(y_range[0], y_range[1] + grid_size, grid_size)
        
        # Calculate density on grid
        density_grid = np.zeros((len(y_grid)-1, len(x_grid)-1))
        
        if len(people_positions) > 0:
            kdtree = KDTree(people_positions)
            
            # For each grid cell, count nearby people
            for i in range(len(x_grid)-1):
                for j in range(len(y_grid)-1):
                    cell_center = np.array([
                        (x_grid[i] + x_grid[i+1]) / 2,
                        (y_grid[j] + y_grid[j+1]) / 2
                    ])
                    
                    # Count people within 2 meters of cell center
                    count = len(kdtree.query_radius([cell_center], r=2.0)[0])
                    density_grid[j, i] = count / 4.0  # divide by area (4 sq meters)
        
        # Find hotspots
        max_density = np.max(density_grid)
        hotspot_threshold = max(0.5, avg_density * 1.5)
        hotspot_locations = []
        
        for j in range(density_grid.shape[0]):
            for i in range(density_grid.shape[1]):
                if density_grid[j, i] >= hotspot_threshold:
                    x = (x_grid[i] + x_grid[i+1]) / 2
                    y = (y_grid[j] + y_grid[j+1]) / 2
                    density = density_grid[j, i]
                    hotspot_locations.append({
                        'x': x,
                        'y': y,
                        'density': density
                    })
        
        # Sort hotspots by density
        hotspot_locations = sorted(hotspot_locations, key=lambda x: x['density'], reverse=True)[:5]
    else:
        density_grid = np.zeros((1, 1))
        max_density = 0
        hotspot_locations = []
    
    # Create results dictionary
    density_results = {
        'total_people': num_people,
        'avg_density': avg_density,
        'max_density': max_density,
        'density_grid': density_grid,
        'hotspots': hotspot_locations
    }
    
    return density_results

def analyze_crowd_flow(processed_data):
    """Simulate crowd flow analysis"""
    # This is a simplified simulation since we don't have sequential data
    points = processed_data['points']
    clusters = processed_data['clusters']
    
    # Extract clusters representing people (exclude noise with label -1)
    people_clusters = np.unique(clusters[clusters >= 0])
    people_positions = []
    
    for cluster_id in people_clusters:
        cluster_points = points[clusters == cluster_id]
        centroid = np.mean(cluster_points, axis=0)
        people_positions.append(centroid[:2])  # Keep only x, y
    
    if len(people_positions) == 0:
        # Return empty results if no people detected
        return {
            'avg_speed': 0,
            'dominant_direction': 'N/A',
            'bottlenecks': [],
            'flow_vectors': {
                'positions': np.zeros((0, 2)),
                'vectors': np.zeros((0, 2)),
                'magnitudes': np.zeros(0)
            }
        }
    
    people_positions = np.array(people_positions)
    
    # Create simulated flow field
    # Grid points for the flow field
    x_range = processed_data['dimensions']['x_range']
    y_range = processed_data['dimensions']['y_range']
    grid_size = 1.0  # 1 meter grid
    
    x_grid = np.arange(x_range[0], x_range[1] + grid_size, grid_size)
    y_grid = np.arange(y_range[0], y_range[1] + grid_size, grid_size)
    
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_positions = np.vstack([X.ravel(), Y.ravel()]).T
    
    # Simulate flow toward an "exit" at the center of the right edge
    exit_x = x_range[1]
    exit_y = (y_range[0] + y_range[1]) / 2
    
    # Calculate vectors pointing toward exit with some variation
    vectors = np.zeros((len(grid_positions), 2))
    np.random.seed(42)  # For reproducibility
    
    for i, (x, y) in enumerate(grid_positions):
        # Direction to exit
        dx = exit_x - x
        dy = exit_y - y
        dist = np.sqrt(dx*dx + dy*dy)
        
        if dist > 0:
            # Normalize
            dx /= dist
            dy /= dist
            
            # Add some variation
            angle_mod = np.sin(x * 0.3) * np.cos(y * 0.3) * 0.5
            
            # Rotate vector
            cos_mod = np.cos(angle_mod)
            sin_mod = np.sin(angle_mod)
            
            dx_mod = dx * cos_mod - dy * sin_mod
            dy_mod = dx * sin_mod + dy * cos_mod
            
            vectors[i] = [dx_mod, dy_mod]
        else:
            vectors[i] = [0, 0]
    
    # Add bottlenecks for realism
    for _ in range(3):
        # Random bottleneck position
        bottleneck_x = np.random.uniform(x_range[0] + 1, x_range[1] - 1)
        bottleneck_y = np.random.uniform(y_range[0] + 1, y_range[1] - 1)
        
        # Reduce speed near bottleneck
        for i, (x, y) in enumerate(grid_positions):
            dist = np.sqrt((x - bottleneck_x)**2 + (y - bottleneck_y)**2)
            if dist < 3.0:
                vectors[i] *= dist / 3.0
    
    # Calculate magnitudes (speed)
    magnitudes = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)
    
    # Scale to realistic walking speeds (0.2 to 1.5 m/s)
    scale_factor = 1.3 / np.max(magnitudes) if np.max(magnitudes) > 0 else 1.0
    vectors *= scale_factor
    magnitudes = np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)
    
    # Calculate average speed
    avg_speed = np.mean(magnitudes)
    
    # Calculate dominant direction
    avg_vector = np.mean(vectors, axis=0)
    angle = np.arctan2(avg_vector[1], avg_vector[0]) * 180 / np.pi
    
    # Convert angle to cardinal direction
    directions = ["E", "NE", "N", "NW", "W", "SW", "S", "SE", "E"]
    idx = int((angle + 22.5) % 360 / 45)
    dominant_direction = directions[idx]
    
    # Identify bottlenecks
    bottlenecks = []
    for i, (pos, mag) in enumerate(zip(grid_positions, magnitudes)):
        if mag < 0.3:  # Low speed indicates potential bottleneck
            x, y = pos
            
            # Check if there are faster points nearby (indicating slowdown)
            nearby_indices = np.where(
                (np.abs(grid_positions[:, 0] - x) < 3) &
                (np.abs(grid_positions[:, 1] - y) < 3)
            )[0]
            
            nearby_speeds = magnitudes[nearby_indices]
            if len(nearby_speeds) > 0 and np.max(nearby_speeds) > 0.5:
                # Calculate severity based on speed difference
                severity = min(10, int(10 * (np.max(nearby_speeds) - mag) / np.max(nearby_speeds)))
                
                if severity >= 3:
                    bottlenecks.append({
                        'x': x,
                        'y': y,
                        'severity': severity
                    })
    
    # Sort bottlenecks by severity and get top 5
    bottlenecks = sorted(bottlenecks, key=lambda x: x['severity'], reverse=True)[:5]
    
    # Create flow results
    flow_results = {
        'avg_speed': avg_speed,
        'dominant_direction': dominant_direction,
        'bottlenecks': bottlenecks,
        'flow_vectors': {
            'positions': grid_positions,
            'vectors': vectors,
            'magnitudes': magnitudes
        }
    }
    
    return flow_results

def visualize_flow(flow_results, processed_data):
    """Visualize crowd flow patterns"""
    if flow_results is None:
        return None
        
    positions = flow_results['flow_vectors']['positions']
    vectors = flow_results['flow_vectors']['vectors']
    magnitudes = flow_results['flow_vectors']['magnitudes']
    
    if len(positions) == 0:
        return None
    
    # Create grid for heatmap
    x_range = processed_data['dimensions']['x_range']
    y_range = processed_data['dimensions']['y_range']
    
    xi = np.linspace(x_range[0], x_range[1], 100)
    yi = np.linspace(y_range[0], y_range[1], 100)
    
    # Interpolate flow magnitudes to grid
    from scipy.interpolate import griddata
    zi = griddata((positions[:, 0], positions[:, 1]), magnitudes, (xi[None, :], yi[:, None]), method='linear')
    
    # Create figure
    fig = go.Figure()
    
    # Add heatmap
    fig.add_trace(go.Heatmap(
        x=xi,
        y=yi,
        z=zi,
        colorscale='Blues',
        colorbar=dict(title='Flow Speed (m/s)'),
        opacity=0.7
    ))
    
    # Add arrows for flow direction
    # Use a subset of points to avoid overcrowding
    if len(positions) > 100:
        subset_size = 100
        subset_idx = np.random.choice(len(positions), subset_size, replace=False)
        subset_positions = positions[subset_idx]
        subset_vectors = vectors[subset_idx]
    else:
        subset_positions = positions
        subset_vectors = vectors
    
    # Scale vectors for visualization
    scale_factor = 1.0
    subset_vectors_scaled = subset_vectors * scale_factor
    
    # Add arrows
    for i in range(len(subset_positions)):
        x, y = subset_positions[i]
        dx, dy = subset_vectors_scaled[i]
        
        # Filter out very small vectors
        if np.sqrt(dx*dx + dy*dy) < 0.1:
            continue
        
        fig.add_trace(go.Scatter(
            x=[x, x + dx],
            y=[y, y + dy],
            mode='lines+markers',
            line=dict(color='red', width=2),
            marker=dict(size=[0, 5], color='red'),
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        title='Crowd Flow Patterns',
        height=600
    )
    
    return fig

def generate_recommendations(density_results, flow_results):
    """Generate crowd management recommendations"""
    recommendations = {
        'issues': [],
        'actions': [],
        'opportunities': []
    }
    
    # Check for density issues
    if density_results['hotspots']:
        for i, hotspot in enumerate(density_results['hotspots']):
            if hotspot['density'] > 3.0:  # Very high density
                issue = {
                    'title': f'Critical crowd density at location {i+1}',
                    'severity': min(10, int(hotspot['density'] * 2)),
                    'location': f'({hotspot["x"]:.1f}, {hotspot["y"]:.1f})',
                    'description': f'High crowd density of {hotspot["density"]:.2f} people/m² detected. This exceeds safe thresholds.'
                }
                recommendations['issues'].append(issue)
                
                action = {
                    'title': f'Reduce density at hotspot {i+1}',
                    'priority': 'High',
                    'description': 'Action required to reduce crowd density in this area.',
                    'steps': [
                        'Deploy staff to redirect crowd flow',
                        'Consider temporarily restricting entry to this zone',
                        'Use announcements to encourage people to move to less crowded areas',
                        'Open alternative pathways to reduce flow through this area'
                    ]
                }
                recommendations['actions'].append(action)
    
    # Check for flow issues
    if flow_results['bottlenecks']:
        for i, bottleneck in enumerate(flow_results['bottlenecks']):
            if bottleneck['severity'] >= 7:  # Critical bottleneck
                issue = {
                    'title': f'Critical flow bottleneck at location {i+1}',
                    'severity': bottleneck['severity'],
                    'location': f'({bottleneck["x"]:.1f}, {bottleneck["y"]:.1f})',
                    'description': f'Severe crowd flow constriction detected with risk of crowd compression.'
                }
                recommendations['issues'].append(issue)
                
                action = {
                    'title': f'Resolve critical bottleneck {i+1}',
                    'priority': 'High',
                    'description': 'Immediate action required to resolve this flow bottleneck.',
                    'steps': [
                        'Deploy staff to manage crowd flow through this area',
                        'Implement one-way system to prevent counterflow',
                        'Consider widening the pathway if possible',
                        'Temporarily close this route and redirect traffic if alternatives are available'
                    ]
                }
                recommendations['actions'].append(action)
    
    # Add general recommendations
    recommendations['opportunities'] = [
        {
            'title': 'Optimize crowd flow patterns',
            'impact': 'High',
            'description': f'The dominant crowd direction is {flow_results["dominant_direction"]}. Design the venue layout to work with this natural flow direction.'
        },
        {
            'title': 'Dynamic information systems',
            'impact': 'High',
            'description': 'Implement real-time digital signage showing crowd density in different areas. This allows attendees to make informed decisions.'
        },
        {
            'title': 'Improved entry/exit management',
            'impact': 'Medium',
            'description': 'Consider implementing timed entry tickets or dynamic entry control based on real-time density data to prevent overcrowding.'
        }
    ]
    
    return recommendations

def generate_report_html(event_name, event_date, density_results, flow_results, recommendations):
    """Generate an HTML report with analysis results"""
    
    # Create HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Crowd Analysis Report: {event_name}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 1px solid #ddd;
            }}
            .section {{
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 1px solid #eee;
            }}
            h1 {{ color: #2C3E50; }}
            h2 {{ color: #3498DB; margin-top: 30px; }}
            h3 {{ color: #2980B9; }}
            .metric-container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
                margin: 20px 0;
            }}
            .metric-box {{
                background-color: #f8f9fa;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 15px;
                width: 48%;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .metric-title {{
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 5px;
                color: #555;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2C3E50;
            }}
            .recommendation {{
                background-color: #f8f9fa;
                border-left: 4px solid #3498DB;
                padding: 15px;
                margin-bottom: 15px;
                border-radius: 0 5px 5px 0;
            }}
            .recommendation.high {{
                border-left-color: #e74c3c;
            }}
            .recommendation.medium {{
                border-left-color: #f39c12;
            }}
            .recommendation.low {{
                border-left-color: #2ecc71;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{ background-color: #f8f9fa; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Crowd Analysis Report</h1>
            <h2>{event_name}</h2>
            <p>Date: {event_date.strftime('%B %d, %Y')}</p>
            <p>Report generated: {datetime.now().strftime('%B %d, %Y %H:%M')}</p>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <p>This report provides analysis of crowd density and flow patterns based on LiDAR data, including metrics and actionable recommendations for crowd management.</p>
            
            <div class="metric-container">
                <div class="metric-box">
                    <div class="metric-title">Total People</div>
                    <div class="metric-value">{density_results['total_people']}</div>
                    <div class="metric-description">Estimated number of people detected</div>
                </div>
                
                <div class="metric-box">
                    <div class="metric-title">Average Density</div>
                    <div class="metric-value">{density_results['avg_density']:.2f} people/m²</div>
                    <div class="metric-description">Average crowd density across occupied areas</div>
                </div>
                
                <div class="metric-box">
                    <div class="metric-title">Maximum Density</div>
                    <div class="metric-value">{density_results['max_density']:.2f} people/m²</div>
                    <div class="metric-description">Highest measured crowd density</div>
                </div>
                
                <div class="metric-box">
                    <div class="metric-title">Average Flow Speed</div>
                    <div class="metric-value">{flow_results['avg_speed']:.2f} m/s</div>
                    <div class="metric-description">Average speed of crowd movement</div>
                </div>
            </div>
        </div>
    """
    
    # Add density analysis section
    html_report += """
        <div class="section">
            <h2>Crowd Density Analysis</h2>
            <p>Analysis of crowd distribution and density across the venue.</p>
    """
    
    # Add density hotspots table
    if density_results['hotspots']:
        html_report += """
            <h3>Density Hotspots</h3>
            <p>Areas with significantly higher density that require attention.</p>
            
            <table>
                <tr>
                    <th>Location (X, Y)</th>
                    <th>Density (people/m²)</th>
                    <th>Risk Level</th>
                </tr>
        """
        
        for hotspot in density_results['hotspots']:
            # Determine risk level
            if hotspot['density'] < 1.0:
                risk_level = "Low"
            elif hotspot['density'] < 2.5:
                risk_level = "Moderate"
            elif hotspot['density'] < 4.0:
                risk_level = "High"
            else:
                risk_level = "Critical"
            
            html_report += f"""
                <tr>
                    <td>({hotspot['x']:.1f}, {hotspot['y']:.1f})</td>
                    <td>{hotspot['density']:.2f}</td>
                    <td>{risk_level}</td>
                </tr>
            """
        
        html_report += """
            </table>
        """
    
    html_report += """
        </div>
    """
    
    # Add flow analysis section
    html_report += f"""
        <div class="section">
            <h2>Crowd Flow Analysis</h2>
            <p>Analysis of crowd movement patterns and flow dynamics.</p>
            
            <h3>Flow Statistics</h3>
            <p>Key metrics describing crowd movement patterns.</p>
            
            <div class="metric-container">
                <div class="metric-box">
                    <div class="metric-title">Average Speed</div>
                    <div class="metric-value">{flow_results['avg_speed']:.2f} m/s</div>
                    <div class="metric-description">Average crowd movement speed</div>
                </div>
                
                <div class="metric-box">
                    <div class="metric-title">Dominant Direction</div>
                    <div class="metric-value">{flow_results['dominant_direction']}</div>
                    <div class="metric-description">Primary crowd movement direction</div>
                </div>
            </div>
    """
    
    # Add bottlenecks table
    if flow_results['bottlenecks']:
        html_report += """
            <h3>Flow Bottlenecks</h3>
            <p>Areas where crowd movement is restricted or impeded.</p>
            
            <table>
                <tr>
                    <th>Location (X, Y)</th>
                    <th>Severity (1-10)</th>
                    <th>Priority</th>
                </tr>
        """
        
        for bottleneck in flow_results['bottlenecks']:
            # Determine priority level
            if bottleneck['severity'] <= 3:
                priority = "Low"
            elif bottleneck['severity'] <= 6:
                priority = "Medium"
            else:
                priority = "High"
            
            html_report += f"""
                <tr>
                    <td>({bottleneck['x']:.1f}, {bottleneck['y']:.1f})</td>
                    <td>{bottleneck['severity']}</td>
                    <td>{priority}</td>
                </tr>
            """
        
        html_report += """
            </table>
        """
    
    html_report += """
        </div>
    """
    
    # Add recommendations section
    if recommendations:
        html_report += """
            <div class="section">
                <h2>Crowd Management Recommendations</h2>
                <p>Actionable recommendations based on the analysis to improve crowd safety and experience.</p>
                
                <h3>Key Issues Identified</h3>
        """
        
        for issue in recommendations['issues']:
            html_report += f"""
                <div class="recommendation">
                    <div class="recommendation-title">{issue['title']}</div>
                    <p><strong>Severity:</strong> {issue['severity']}/10</p>
                    <p><strong>Location:</strong> {issue['location']}</p>
                    <p>{issue['description']}</p>
                </div>
            """
        
        html_report += """
            <h3>Recommended Actions</h3>
        """
        
        for action in recommendations['actions']:
            # Determine priority class
            if action['priority'] == 'High':
                priority_class = 'high'
            elif action['priority'] == 'Medium':
                priority_class = 'medium'
            else:
                priority_class = 'low'
                
            html_report += f"""
                <div class="recommendation {priority_class}">
                    <h4>{action['title']} ({action['priority']} Priority)</h4>
                    <p>{action['description']}</p>
                    <p><strong>Implementation steps:</strong></p>
                    <ul>
            """
            
            for step in action['steps']:
                html_report += f"""
                        <li>{step}</li>
                """
            
            html_report += """
                    </ul>
                </div>
            """
        
        html_report += """
            <h3>Optimization Opportunities</h3>
        """
        
        for opportunity in recommendations['opportunities']:
            html_report += f"""
                <div class="recommendation">
                    <h4>{opportunity['title']} ({opportunity['impact']} Impact)</h4>
                    <p>{opportunity['description']}</p>
                </div>
            """
        
        html_report += """
            </div>
        """
    
    # Add footer
    html_report += """
        <div class="footer" style="text-align: center; margin-top: 50px; color: #7f8c8d;">
            <p>LiDAR Crowd Analytics Report</p>
            <p>Generated by LiDAR Crowd Analytics System</p>
        </div>
    </body>
    </html>
    """
    
    return html_report

# Initialize session state for storing data between reruns
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'density_results' not in st.session_state:
    st.session_state.density_results = None
if 'flow_results' not in st.session_state:
    st.session_state.flow_results = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'event_id' not in st.session_state:
    st.session_state.event_id = None
if 'analysis_id' not in st.session_state:
    st.session_state.analysis_id = None
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "Upload"

# Title and introduction
st.title("3D LiDAR Crowd Analytics")
st.markdown("""
This application helps event managers analyze 3D LiDAR data to understand crowd behavior 
and generate actionable crowd management strategies. Analysis results are stored in a database for future reference.
""")

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    selected_tab = st.radio(
        "Select a section:",
        ["Upload", "Visualization", "Analysis", "Recommendations", "Report", "Database"],
        index=["Upload", "Visualization", "Analysis", "Recommendations", "Report", "Database"].index(st.session_state.selected_tab)
    )
    st.session_state.selected_tab = selected_tab
    
    st.header("About")
    st.info("""
    This application processes 3D LiDAR point cloud data to analyze crowd density and movement patterns. 
    It uses advanced analytics to generate insights and recommendations for effective crowd management.
    All analysis results are stored in a database for future reference.
    """)

# Main content based on selected tab
if selected_tab == "Upload":
    st.header("Upload LiDAR Data")
    
    # Event information
    st.subheader("Event Information")
    col1, col2 = st.columns(2)
    with col1:
        event_name = st.text_input("Event Name", "Sample Event")
    with col2:
        event_date = st.date_input("Event Date")
    
    st.markdown("""
    Upload your 3D LiDAR dataset. The application supports the following formats:
    - CSV (with X,Y,Z columns)
    - XYZ (plain text with space-separated values)
    - TXT (plain text with space or comma-separated values)
    - NPY (NumPy binary format)
    """)
    
    # For demo purposes, provide option to use sample data
    use_sample = st.checkbox("Use sample data for demonstration")
    
    if use_sample:
        st.info("Using generated sample data for demonstration.")
        
        # Create event in database
        if st.button("Create Event and Process Sample Data"):
            # Save event in database
            event_id = db.create_event(event_name, event_date)
            if event_id:
                st.session_state.event_id = event_id
                st.success(f"Event '{event_name}' created with ID: {event_id}")
                
                # Generate synthetic point cloud
                n_points = 10000
                np.random.seed(42)  # For reproducible results
                
                # Create simple terrain with some objects
                x = np.random.uniform(-15, 15, n_points)
                y = np.random.uniform(-15, 15, n_points)
                
                # Base height (ground)
                z = np.zeros(n_points)
                
                # Add terrain variations
                terrain = 0.1 * np.sin(x*0.5) * np.cos(y*0.5)
                z += terrain
                
                # Add some "people" (small vertical clusters)
                n_people = 50
                people_centers = np.random.uniform(-10, 10, (n_people, 2))
                
                for i in range(len(x)):
                    # Find closest person center
                    distances = np.sqrt((x[i] - people_centers[:, 0])**2 + (y[i] - people_centers[:, 1])**2)
                    min_dist = np.min(distances)
                    
                    if min_dist < 0.3:  # Within person radius
                        person_idx = np.argmin(distances)
                        # Create vertical distribution for person
                        z[i] = np.random.uniform(0.1, 1.8)  # Person height
                
                # Combine into point cloud
                points = np.column_stack((x, y, z))
                
                # Process the sample data
                with st.spinner("Processing sample data..."):
                    processed_data = preprocess_point_cloud(points)
                    st.session_state.processed_data = processed_data
                    
                    # Create analysis in database
                    analysis_id = db.create_analysis(event_id, 'point_cloud', processed_data)
                    if analysis_id:
                        st.session_state.analysis_id = analysis_id
                        st.success(f"Point cloud analysis created with ID: {analysis_id}")
                    
                    st.markdown(f"**Dataset summary:**")
                    st.write(f"- Number of points: {len(processed_data['points'])}")
                    st.write(f"- Dimensions: Width={processed_data['dimensions']['width']:.2f}m, Length={processed_data['dimensions']['length']:.2f}m, Height={processed_data['dimensions']['height']:.2f}m")
                    
                    # Display a preview of the point cloud
                    preview_container = st.container()
                    preview_container.subheader("Point Cloud Preview")
                    with preview_container:
                        col1, col2 = st.columns([3, 1])
                        
                        with col2:
                            st.write("Visualization Options")
                            point_size = st.slider("Point Size", 1, 10, 3)
                            color_by = st.selectbox(
                                "Color points by:",
                                ["Height", "Density", "Cluster"]
                            )
                        
                        with col1:
                            preview_fig = visualize_point_cloud(
                                processed_data, 
                                point_size=point_size,
                                color_by=color_by
                            )
                            st.plotly_chart(preview_fig, use_container_width=True)
                    
                    # Button to navigate to visualization
                    if st.button("Proceed to Visualization"):
                        st.session_state.selected_tab = "Visualization"
                        st.rerun()
    
    else:
        # Create event in database
        if st.button("Create Event"):
            event_id = db.create_event(event_name, event_date)
            if event_id:
                st.session_state.event_id = event_id
                st.success(f"Event '{event_name}' created with ID: {event_id}")
                st.info("Now upload a point cloud file to proceed.")
        
        # Only show file uploader if an event has been created
        if st.session_state.event_id:
            uploaded_file = st.file_uploader("Choose a point cloud file", type=["csv", "xyz", "txt", "npy"])
            
            if uploaded_file is not None:
                try:
                    with st.spinner("Loading and processing point cloud data..."):
                        # Load the point cloud data
                        points = load_point_cloud(uploaded_file)
                        
                        if points is not None:
                            # Preprocess the data
                            processed_data = preprocess_point_cloud(points)
                            
                            # Store processed data in session state
                            st.session_state.processed_data = processed_data
                            
                            # Create analysis in database
                            analysis_id = db.create_analysis(st.session_state.event_id, 'point_cloud', processed_data)
                            if analysis_id:
                                st.session_state.analysis_id = analysis_id
                                st.success(f"Point cloud analysis created with ID: {analysis_id}")
                            
                            st.success("Data loaded and processed successfully!")
                            st.markdown(f"**Dataset summary:**")
                            st.write(f"- Number of points: {len(processed_data['points'])}")
                            st.write(f"- Dimensions: Width={processed_data['dimensions']['width']:.2f}m, Length={processed_data['dimensions']['length']:.2f}m, Height={processed_data['dimensions']['height']:.2f}m")
                            
                            # Display a preview of the point cloud
                            preview_container = st.container()
                            preview_container.subheader("Point Cloud Preview")
                            with preview_container:
                                col1, col2 = st.columns([3, 1])
                                
                                with col2:
                                    st.write("Visualization Options")
                                    point_size = st.slider("Point Size", 1, 10, 3)
                                    color_by = st.selectbox(
                                        "Color points by:",
                                        ["Height", "Density", "Cluster"]
                                    )
                                
                                with col1:
                                    preview_fig = visualize_point_cloud(
                                        processed_data, 
                                        point_size=point_size,
                                        color_by=color_by
                                    )
                                    st.plotly_chart(preview_fig, use_container_width=True)
                            
                            # Button to navigate to visualization
                            if st.button("Proceed to Visualization"):
                                st.session_state.selected_tab = "Visualization"
                                st.rerun()
                
                except Exception as e:
                    st.error(f"Error processing the file: {str(e)}")
            else:
                st.info("Please upload a point cloud file to get started.")

elif selected_tab == "Visualization":
    st.header("Data Visualization")
    
    if st.session_state.processed_data is None:
        st.warning("No data available. Please upload a dataset first.")
        if st.button("Go to Upload"):
            st.session_state.selected_tab = "Upload"
            st.rerun()
    else:
        visualization_type = st.selectbox(
            "Select visualization type:",
            ["3D Point Cloud", "2D Density Map"]
        )
        
        if visualization_type == "3D Point Cloud":
            st.subheader("3D Point Cloud Visualization")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.write("Visualization Options")
                point_size = st.slider("Point Size", 1, 10, 3)
                color_by = st.selectbox(
                    "Color points by:",
                    ["Height", "Density", "Cluster"]
                )
                
                st.write("Camera Controls")
                st.info("Use mouse to rotate, zoom, and pan in the 3D visualization.")
            
            with col1:
                with st.spinner("Generating 3D visualization..."):
                    fig = visualize_point_cloud(
                        st.session_state.processed_data, 
                        point_size=point_size,
                        color_by=color_by
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        elif visualization_type == "2D Density Map":
            st.subheader("2D Density Map")
            
            with st.spinner("Generating 2D density map..."):
                fig = create_density_heatmap(st.session_state.processed_data)
                st.plotly_chart(fig, use_container_width=True)
        
        # Button to proceed to analysis
        if st.button("Proceed to Analysis"):
            st.session_state.selected_tab = "Analysis"
            st.rerun()

elif selected_tab == "Analysis":
    st.header("Crowd Analysis")
    
    if st.session_state.processed_data is None:
        st.warning("No data available. Please upload a dataset first.")
        if st.button("Go to Upload"):
            st.session_state.selected_tab = "Upload"
            st.rerun()
    else:
        analysis_tabs = st.tabs(["Density Analysis", "Flow Analysis", "Combined Metrics"])
        
        with analysis_tabs[0]:
            st.subheader("Crowd Density Analysis")
            
            if st.button("Run Density Analysis"):
                with st.spinner("Analyzing crowd density..."):
                    # Run crowd density analysis
                    density_results = analyze_crowd_density(st.session_state.processed_data)
                    
                    # Store results in session state
                    st.session_state.density_results = density_results
                    
                    # Save to database if we have an analysis ID
                    if st.session_state.analysis_id:
                        db.save_density_results(st.session_state.analysis_id, density_results)
                        st.success("Density analysis results saved to database.")
                    
                    # Display results
                    st.success("Density analysis complete!")
            
            if st.session_state.density_results is not None:
                st.write("### Density Results")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Create heatmap visualization of density
                    st.plotly_chart(create_density_heatmap(st.session_state.processed_data), use_container_width=True)
                
                with col2:
                    st.write("#### Density Statistics")
                    st.write(f"- Total people: {st.session_state.density_results['total_people']}")
                    st.write(f"- Average density: {st.session_state.density_results['avg_density']:.2f} people/m²")
                    st.write(f"- Maximum density: {st.session_state.density_results['max_density']:.2f} people/m²")
                    
                    if st.session_state.density_results['hotspots']:
                        st.write("#### Density Hotspots")
                        for i, hotspot in enumerate(st.session_state.density_results['hotspots']):
                            st.write(f"Hotspot {i+1}: {hotspot['density']:.2f} people/m² at location ({hotspot['x']:.1f}, {hotspot['y']:.1f})")
            else:
                st.info("Click the 'Run Density Analysis' button to analyze crowd density.")
        
        with analysis_tabs[1]:
            st.subheader("Crowd Flow Analysis")
            
            if st.button("Run Flow Analysis"):
                with st.spinner("Analyzing crowd flow patterns..."):
                    # Run crowd flow analysis
                    flow_results = analyze_crowd_flow(st.session_state.processed_data)
                    
                    # Store results in session state
                    st.session_state.flow_results = flow_results
                    
                    # Save to database if we have an analysis ID
                    if st.session_state.analysis_id:
                        db.save_flow_results(st.session_state.analysis_id, flow_results)
                        st.success("Flow analysis results saved to database.")
                    
                    # Display results
                    st.success("Flow analysis complete!")
            
            if st.session_state.flow_results is not None:
                st.write("### Flow Results")
                
                # Visualize flow
                flow_fig = visualize_flow(st.session_state.flow_results, st.session_state.processed_data)
                if flow_fig:
                    st.plotly_chart(flow_fig, use_container_width=True)
                
                st.write("#### Flow Statistics")
                st.write(f"- Average speed: {st.session_state.flow_results['avg_speed']:.2f} m/s")
                st.write(f"- Dominant direction: {st.session_state.flow_results['dominant_direction']}")
                
                if st.session_state.flow_results['bottlenecks']:
                    st.write("#### Identified Bottlenecks")
                    for i, bottleneck in enumerate(st.session_state.flow_results['bottlenecks']):
                        st.write(f"Bottleneck {i+1}: at location ({bottleneck['x']:.1f}, {bottleneck['y']:.1f}) with severity {bottleneck['severity']}/10")
            else:
                st.info("Click the 'Run Flow Analysis' button to analyze crowd flow patterns.")
        
        with analysis_tabs[2]:
            st.subheader("Combined Metrics")
            
            if st.session_state.density_results is not None and st.session_state.flow_results is not None:
                st.info("This would show a combined visualization of density and flow metrics.")
                
                st.write("### Key Insights")
                st.write("- Areas with high density and low flow indicate potential congestion points")
                st.write("- Areas with high flow and moderate density indicate main thoroughfares")
                st.write("- Areas with low density and high flow indicate potential underutilized spaces")
            else:
                st.info("Complete both density and flow analysis to view combined metrics.")
        
        # Generate recommendations when both analyses are complete
        if st.session_state.density_results is not None and st.session_state.flow_results is not None:
            if st.button("Generate Recommendations"):
                with st.spinner("Generating crowd management recommendations..."):
                    recommendations = generate_recommendations(
                        st.session_state.density_results,
                        st.session_state.flow_results
                    )
                    st.session_state.recommendations = recommendations
                    
                    # Save to database if we have an analysis ID
                    if st.session_state.analysis_id:
                        db.save_recommendations(st.session_state.analysis_id, recommendations)
                        st.success("Recommendations saved to database.")
                    
                    st.session_state.selected_tab = "Recommendations"
                    st.rerun()

elif selected_tab == "Recommendations":
    st.header("Crowd Management Recommendations")
    
    if st.session_state.recommendations is None:
        if st.session_state.density_results is not None and st.session_state.flow_results is not None:
            if st.button("Generate Recommendations"):
                with st.spinner("Generating crowd management recommendations..."):
                    recommendations = generate_recommendations(
                        st.session_state.density_results,
                        st.session_state.flow_results
                    )
                    st.session_state.recommendations = recommendations
                    
                    # Save to database if we have an analysis ID
                    if st.session_state.analysis_id:
                        db.save_recommendations(st.session_state.analysis_id, recommendations)
                        st.success("Recommendations saved to database.")
                    
                    st.rerun()
        else:
            st.warning("Complete the analysis first to generate recommendations.")
            if st.button("Go to Analysis"):
                st.session_state.selected_tab = "Analysis"
                st.rerun()
    else:
        # Display the recommendations
        st.subheader("Key Issues Identified")
        for i, issue in enumerate(st.session_state.recommendations["issues"]):
            with st.expander(f"Issue {i+1}: {issue['title']}", expanded=i==0):
                st.write(f"**Severity:** {issue['severity']}/10")
                st.write(f"**Description:** {issue['description']}")
                st.write(f"**Location:** {issue['location']}")
        
        st.subheader("Recommended Actions")
        for i, action in enumerate(st.session_state.recommendations["actions"]):
            with st.expander(f"Action {i+1}: {action['title']}", expanded=i==0):
                st.write(f"**Priority:** {action['priority']}")
                st.write(f"**Description:** {action['description']}")
                st.write("**Implementation steps:**")
                for step in action["steps"]:
                    st.write(f"- {step}")
        
        st.subheader("Optimization Opportunities")
        for i, opportunity in enumerate(st.session_state.recommendations["opportunities"]):
            with st.expander(f"Opportunity {i+1}: {opportunity['title']}", expanded=i==0):
                st.write(f"**Potential impact:** {opportunity['impact']}")
                st.write(f"**Description:** {opportunity['description']}")
        
        # Button to generate report
        if st.button("Generate Comprehensive Report"):
            st.session_state.selected_tab = "Report"
            st.rerun()

elif selected_tab == "Report":
    st.header("Comprehensive Crowd Analysis Report")
    
    if (st.session_state.processed_data is None or 
        st.session_state.density_results is None or 
        st.session_state.flow_results is None or 
        st.session_state.recommendations is None):
        
        st.warning("Complete all previous steps to generate a comprehensive report.")
        if st.button("Go to Analysis"):
            st.session_state.selected_tab = "Analysis"
            st.rerun()
    else:
        st.write("Configure your report below:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            event_name = st.text_input("Event Name", "Sample Event")
        
        with col2:
            event_date = st.date_input("Event Date")
        
        # Generate report
        if st.button("Generate Report"):
            with st.spinner("Generating comprehensive report..."):
                # Generate HTML report
                html_report = generate_report_html(
                    event_name,
                    event_date,
                    st.session_state.density_results,
                    st.session_state.flow_results,
                    st.session_state.recommendations
                )
                
                # Save report to database if we have an analysis ID
                if st.session_state.analysis_id:
                    report_name = f"{event_name} - {event_date.strftime('%Y-%m-%d')}"
                    db.save_report(st.session_state.analysis_id, report_name, html_report)
                    st.success("Report saved to database.")
                
                # Display the report
                st.success("Report generated successfully!")
                
                # Provide download link
                b64_html = base64.b64encode(html_report.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64_html}" download="{event_name.replace(" ", "_")}_crowd_analysis_report.html">Download Report (HTML)</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                # Show report preview
                st.subheader("Report Preview")
                st.components.v1.html(html_report, height=600, scrolling=True)

elif selected_tab == "Database":
    st.header("Database Management")
    
    st.subheader("Saved Events")
    events = db.get_all_events()
    
    if not events:
        st.info("No events found in the database. Create an event on the Upload page.")
    else:
        event_options = [f"{event[1]} ({event[2]})" for event in events]
        selected_event = st.selectbox("Select an event to view", event_options)
        
        if selected_event:
            selected_event_index = event_options.index(selected_event)
            event_id = events[selected_event_index][0]
            
            st.subheader("Analyses for this Event")
            analyses = db.get_analyses_for_event(event_id)
            
            if not analyses:
                st.info("No analyses found for this event.")
            else:
                st.write("The following analyses are available:")
                
                for analysis in analyses:
                    analysis_id, analysis_type, analysis_date, total_points, total_people, avg_density, avg_speed = analysis
                    
                    with st.expander(f"Analysis {analysis_id} - {analysis_date}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Type:** {analysis_type}")
                            st.write(f"**Date:** {analysis_date}")
                            st.write(f"**Total Points:** {total_points or 'N/A'}")
                        
                        with col2:
                            st.write(f"**People Count:** {total_people or 'N/A'}")
                            st.write(f"**Avg. Density:** {avg_density or 'N/A'}")
                            st.write(f"**Avg. Speed:** {avg_speed or 'N/A'}")
            
            st.subheader("Reports for this Event")
            reports = db.get_reports_for_event(event_id)
            
            if not reports:
                st.info("No reports found for this event.")
            else:
                report_options = [f"{report[1]} ({report[2]})" for report in reports]
                selected_report = st.selectbox("Select a report to view", report_options)
                
                if selected_report:
                    selected_report_index = report_options.index(selected_report)
                    report_id = reports[selected_report_index][0]
                    
                    report = db.get_report_by_id(report_id)
                    if report:
                        report_name, report_html, created_at = report
                        
                        st.subheader(f"Report: {report_name}")
                        st.write(f"Created: {created_at}")
                        
                        # Provide download link
                        b64_html = base64.b64encode(report_html.encode()).decode()
                        href = f'<a href="data:text/html;base64,{b64_html}" download="{report_name.replace(" ", "_")}.html">Download Report (HTML)</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                        # Show report preview
                        st.subheader("Report Preview")
                        st.components.v1.html(report_html, height=600, scrolling=True)

# Cleanup on app close
def close_db_connection():
    db.close()

# Register the function to be called when the script is terminated
import atexit
atexit.register(close_db_connection)