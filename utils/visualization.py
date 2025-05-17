import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from sklearn.neighbors import KDTree
from scipy.spatial import cKDTree

def visualize_point_cloud(processed_data, point_size=3, color_by="Height", show_grid=True, preview=False):
    """
    Create an interactive 3D visualization of the LiDAR point cloud.
    
    Args:
        processed_data (dict): Processed point cloud data
        point_size (int): Size of the points in the visualization
        color_by (str): How to color the points ('Height', 'Density', 'Distance from Center', 'Cluster')
        show_grid (bool): Whether to show the grid in the visualization
        preview (bool): If True, creates a simplified preview for faster rendering
        
    Returns:
        plotly.graph_objects.Figure: Interactive 3D figure
    """
    # Extract data
    points = processed_data['points']
    clusters = processed_data['clusters']
    
    # Downsample for preview or if there are too many points
    if preview or len(points) > 50000:
        downsample_factor = 0.05 if preview else 0.2
        indices = np.random.choice(len(points), int(len(points) * downsample_factor), replace=False)
        points = points[indices]
        clusters = clusters[indices] if clusters is not None else None
    
    # Prepare colors based on the selected method
    if color_by == "Height":
        colors = points[:, 2]  # Z-coordinate for height
        colorscale = 'Viridis'
        colorbar_title = 'Height (m)'
    
    elif color_by == "Density":
        # Calculate local density using KD-tree
        kdtree = KDTree(points)
        # Count neighbors within 0.5m radius
        densities = kdtree.query_radius(points, r=0.5, count_only=True)
        colors = densities
        colorscale = 'Reds'
        colorbar_title = 'Local Point Density'
    
    elif color_by == "Distance from Center":
        # Calculate centroid
        centroid = np.mean(points, axis=0)
        # Calculate Euclidean distance from centroid
        distances = np.sqrt(np.sum((points - centroid)**2, axis=1))
        colors = distances
        colorscale = 'Blues'
        colorbar_title = 'Distance from Center (m)'
    
    elif color_by == "Cluster":
        colors = clusters
        colorscale = 'Rainbow'
        colorbar_title = 'Cluster ID'
    
    # Create scatter3d trace
    scatter = go.Scatter3d(
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
    )
    
    # Create figure
    fig = go.Figure(data=[scatter])
    
    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            xaxis=dict(showgrid=show_grid),
            yaxis=dict(showgrid=show_grid),
            zaxis=dict(showgrid=show_grid)
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600
    )
    
    return fig

def create_density_heatmap(processed_data, density_data=None, projection_dims=["x", "y"], resolution=100, as_heatmap=True):
    """
    Create a heatmap visualization of crowd density.
    
    Args:
        processed_data (dict): Processed point cloud data
        density_data (numpy.ndarray, optional): Pre-computed density grid
        projection_dims (list): Which dimensions to use for projection (e.g., ["x", "y"] for top view)
        resolution (int): Resolution of the heatmap grid
        as_heatmap (bool): If True, creates a heatmap, otherwise a scatter plot
        
    Returns:
        plotly.graph_objects.Figure: Heatmap or scatter figure
    """
    points = processed_data['points']
    dimensions = processed_data['dimensions']
    
    # Map dimension names to indices
    dim_map = {"x": 0, "y": 1, "z": 2}
    dim1_idx = dim_map[projection_dims[0]]
    dim2_idx = dim_map[projection_dims[1]]
    
    # Extract projection dimensions
    dim1_data = points[:, dim1_idx]
    dim2_data = points[:, dim2_idx]
    
    if density_data is None:
        # Create density heatmap using histogram2d
        dim1_range = dimensions[f'{projection_dims[0]}_range']
        dim2_range = dimensions[f'{projection_dims[1]}_range']
        
        heatmap, x_edges, y_edges = np.histogram2d(
            dim1_data, dim2_data,
            bins=resolution,
            range=[dim1_range, dim2_range]
        )
        
        # Transpose for correct orientation
        heatmap = heatmap.T
    else:
        # Use provided density data
        heatmap = density_data
        
        # Create axis edges
        dim1_range = dimensions[f'{projection_dims[0]}_range']
        dim2_range = dimensions[f'{projection_dims[1]}_range']
        x_edges = np.linspace(dim1_range[0], dim1_range[1], heatmap.shape[1] + 1)
        y_edges = np.linspace(dim2_range[0], dim2_range[1], heatmap.shape[0] + 1)
    
    # Create axis centers
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    
    # Create figure
    if as_heatmap:
        fig = go.Figure(data=go.Heatmap(
            z=heatmap,
            x=x_centers,
            y=y_centers,
            colorscale='Viridis',
            colorbar=dict(title='Density')
        ))
    else:
        # Create scatter plot with density coloring
        points_2d = np.vstack([dim1_data, dim2_data]).T
        
        # Use KD-tree to estimate local density
        from sklearn.neighbors import KDTree
        kdtree = KDTree(points_2d)
        densities = kdtree.query_radius(points_2d, r=0.5, count_only=True)
        
        fig = go.Figure(data=go.Scatter(
            x=dim1_data,
            y=dim2_data,
            mode='markers',
            marker=dict(
                size=5,
                color=densities,
                colorscale='Viridis',
                colorbar=dict(title='Local Density')
            )
        ))
    
    # Update layout
    fig.update_layout(
        xaxis_title=f'{projection_dims[0]} (m)',
        yaxis_title=f'{projection_dims[1]} (m)',
        title=f'Density Projection ({projection_dims[0]}-{projection_dims[1]})',
        height=600,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

def create_flow_visualization(processed_data, flow_vectors):
    """
    Create a visualization of crowd flow patterns.
    
    Args:
        processed_data (dict): Processed point cloud data
        flow_vectors (dict): Flow vector data containing:
            - positions: array of (x, y) positions
            - vectors: array of (dx, dy) flow vectors
            - magnitudes: array of flow vector magnitudes
        
    Returns:
        plotly.graph_objects.Figure: Flow visualization figure
    """
    positions = flow_vectors['positions']
    vectors = flow_vectors['vectors']
    magnitudes = flow_vectors['magnitudes']
    
    # Create streamline plot
    fig = go.Figure()
    
    # Add a heatmap for the flow magnitude
    x_range = processed_data['dimensions']['x_range']
    y_range = processed_data['dimensions']['y_range']
    
    # Create grid for heatmap
    xi = np.linspace(x_range[0], x_range[1], 100)
    yi = np.linspace(y_range[0], y_range[1], 100)
    
    # Interpolate flow magnitudes to grid
    from scipy.interpolate import griddata
    zi = griddata((positions[:, 0], positions[:, 1]), magnitudes, (xi[None, :], yi[:, None]), method='linear')
    
    # Add heatmap
    fig.add_trace(go.Heatmap(
        x=xi,
        y=yi,
        z=zi,
        colorscale='Blues',
        colorbar=dict(title='Flow Speed (m/s)'),
        opacity=0.7
    ))
    
    # Add arrows to show flow direction
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
        height=600,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

def plot_crowd_metrics(density_results, flow_results):
    """
    Create visualizations of combined crowd metrics.
    
    Args:
        density_results (dict): Results from crowd density analysis
        flow_results (dict): Results from crowd flow analysis
        
    Returns:
        plotly.graph_objects.Figure: Combined metrics visualization
    """
    # Create a subplot with 1 row and 2 columns
    fig = go.Figure()
    
    # Convert data to more convenient format
    density_df = pd.DataFrame({
        'x': density_results['grid_coordinates'][0],
        'y': density_results['grid_coordinates'][1],
        'density': density_results['density_values']
    })
    
    flow_df = pd.DataFrame({
        'x': flow_results['flow_vectors']['positions'][:, 0],
        'y': flow_results['flow_vectors']['positions'][:, 1],
        'speed': flow_results['flow_vectors']['magnitudes']
    })
    
    # Join the datasets on closest x,y coordinates
    from scipy.spatial import cKDTree
    density_points = density_df[['x', 'y']].values
    flow_points = flow_df[['x', 'y']].values
    
    # Find nearest density point for each flow point
    kdtree = cKDTree(density_points)
    distances, indices = kdtree.query(flow_points, k=1)
    
    # Create combined dataframe
    combined_df = flow_df.copy()
    combined_df['density'] = density_df.iloc[indices]['density'].values
    
    # Calculate composite metric (e.g., density/speed ratio)
    # Higher values indicate potential congestion
    combined_df['congestion_risk'] = combined_df['density'] / (combined_df['speed'] + 0.1)  # Add 0.1 to avoid division by zero
    
    # Normalize the congestion risk
    max_risk = combined_df['congestion_risk'].max()
    combined_df['congestion_risk_normalized'] = combined_df['congestion_risk'] / max_risk * 10  # Scale to 0-10
    
    # Create bubble chart
    fig = px.scatter(combined_df, x='x', y='y', 
                     size='density', 
                     color='congestion_risk_normalized',
                     color_continuous_scale='RdYlGn_r',  # Red = high risk, Green = low risk
                     range_color=[0, 10],
                     hover_data=['density', 'speed', 'congestion_risk_normalized'],
                     labels={
                         'x': 'X (m)',
                         'y': 'Y (m)',
                         'density': 'Crowd Density (people/m²)',
                         'speed': 'Movement Speed (m/s)',
                         'congestion_risk_normalized': 'Congestion Risk (0-10)'
                     },
                     title='Crowd Congestion Risk Analysis')
    
    # Add contour lines for density
    x_range = [combined_df['x'].min(), combined_df['x'].max()]
    y_range = [combined_df['y'].min(), combined_df['y'].max()]
    
    # Create grid
    xi = np.linspace(x_range[0], x_range[1], 100)
    yi = np.linspace(y_range[0], y_range[1], 100)
    
    # Interpolate density to grid
    from scipy.interpolate import griddata
    zi = griddata((combined_df['x'], combined_df['y']), combined_df['density'], (xi[None, :], yi[:, None]), method='linear')
    
    # Add contour
    fig.add_trace(go.Contour(
        z=zi,
        x=xi,
        y=yi,
        contours=dict(
            showlabels=True,
            labelfont=dict(size=10, color='white')
        ),
        line=dict(width=0.5, color='white'),
        colorscale='Blues',
        showscale=False,
        opacity=0.3
    ))
    
    # Update layout
    fig.update_layout(
        height=700,
        margin=dict(l=20, r=20, b=20, t=40)
    )
    
    return fig
