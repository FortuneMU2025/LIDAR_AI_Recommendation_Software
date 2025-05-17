import streamlit as st
import numpy as np
import pandas as pd
import os
import tempfile
import io
from utils.data_processing import load_lidar_data, preprocess_lidar_data
from utils.visualization import (
    visualize_point_cloud, 
    create_density_heatmap, 
    create_flow_visualization,
    plot_crowd_metrics
)
from models.crowd_density_model import CrowdDensityModel
from models.crowd_flow_model import CrowdFlowModel
from utils.report_generator import generate_report
from utils.recommendations import generate_recommendations

st.set_page_config(
    page_title="LiDAR Crowd Analytics",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'density_results' not in st.session_state:
    st.session_state.density_results = None
if 'flow_results' not in st.session_state:
    st.session_state.flow_results = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "Upload"

# Title and introduction
st.title("3D LiDAR Crowd Analytics")
st.markdown("""
This application helps event managers analyze 3D LiDAR data to understand crowd behavior 
and generate actionable crowd management strategies.
""")

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    selected_tab = st.radio(
        "Select a section:",
        ["Upload", "Visualization", "Analysis", "Recommendations", "Report"],
        index=["Upload", "Visualization", "Analysis", "Recommendations", "Report"].index(st.session_state.selected_tab)
    )
    st.session_state.selected_tab = selected_tab
    
    st.header("About")
    st.info("""
    This application processes 3D LiDAR point cloud data to analyze crowd density and movement patterns. 
    It uses machine learning to generate insights and recommendations for effective crowd management.
    """)

# Main content based on selected tab
if selected_tab == "Upload":
    st.header("Upload LiDAR Data")
    st.markdown("""
    Upload your 3D LiDAR dataset in PCD format. The application will process the data and prepare it for analysis.
    """)
    
    uploaded_file = st.file_uploader("Choose a PCD file", type=["pcd"])
    
    if uploaded_file is not None:
        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pcd') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        try:
            with st.spinner("Loading and processing LiDAR data..."):
                # Load the LiDAR data
                point_cloud = load_lidar_data(temp_file_path)
                
                # Preprocess the data
                processed_data = preprocess_lidar_data(point_cloud)
                
                # Store processed data in session state
                st.session_state.processed_data = processed_data
                
                st.success("Data loaded and processed successfully!")
                st.markdown(f"**Dataset summary:**")
                st.write(f"- Number of points: {len(processed_data['points'])}")
                st.write(f"- Dimensions: {processed_data['dimensions']}")
                
                # Display a preview of the point cloud
                preview_container = st.container()
                preview_container.subheader("Point Cloud Preview")
                preview_container.write("A simplified preview of your uploaded LiDAR data:")
                preview_fig = visualize_point_cloud(processed_data, preview=True)
                preview_container.plotly_chart(preview_fig, use_container_width=True)
                
                # Button to navigate to visualization
                if st.button("Proceed to Visualization"):
                    st.session_state.selected_tab = "Visualization"
                    st.rerun()
        
        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")
        
        finally:
            # Remove the temporary file
            os.unlink(temp_file_path)
    else:
        st.info("Please upload a PCD file to get started.")

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
            ["3D Point Cloud", "2D Projection"]
        )
        
        if visualization_type == "3D Point Cloud":
            st.subheader("3D Point Cloud Visualization")
            
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.write("Visualization Options")
                point_size = st.slider("Point Size", 1, 10, 3)
                color_by = st.selectbox(
                    "Color points by:",
                    ["Height", "Density", "Distance from Center"]
                )
                show_grid = st.checkbox("Show Grid", value=True)
                
                st.write("Camera Controls")
                st.info("Use mouse to rotate, zoom, and pan in the 3D visualization.")
            
            with col1:
                with st.spinner("Generating 3D visualization..."):
                    fig = visualize_point_cloud(
                        st.session_state.processed_data, 
                        point_size=point_size,
                        color_by=color_by,
                        show_grid=show_grid,
                        preview=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        elif visualization_type == "2D Projection":
            st.subheader("2D Projection")
            
            projection_type = st.selectbox(
                "Select projection type:",
                ["Top View (XY)", "Side View (XZ)", "Front View (YZ)"]
            )
            
            with st.spinner("Generating 2D projection..."):
                if projection_type == "Top View (XY)":
                    projection_dims = ["x", "y"]
                elif projection_type == "Side View (XZ)":
                    projection_dims = ["x", "z"]
                else:  # Front View (YZ)
                    projection_dims = ["y", "z"]
                
                fig = create_density_heatmap(
                    st.session_state.processed_data,
                    projection_dims=projection_dims,
                    resolution=100,
                    as_heatmap=True
                )
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
                    # Initialize and run the crowd density model
                    density_model = CrowdDensityModel()
                    density_results = density_model.analyze(st.session_state.processed_data)
                    
                    # Store results in session state
                    st.session_state.density_results = density_results
                    
                    # Display results
                    st.success("Density analysis complete!")
            
            if st.session_state.density_results is not None:
                st.write("### Density Results")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    density_fig = create_density_heatmap(
                        st.session_state.processed_data,
                        density_data=st.session_state.density_results["density_map"],
                        projection_dims=["x", "y"],
                        resolution=100,
                        as_heatmap=True
                    )
                    st.plotly_chart(density_fig, use_container_width=True)
                
                with col2:
                    st.write("#### Density Statistics")
                    st.write(f"- Average density: {st.session_state.density_results['avg_density']:.2f} people/m²")
                    st.write(f"- Maximum density: {st.session_state.density_results['max_density']:.2f} people/m²")
                    st.write(f"- Total people estimate: {st.session_state.density_results['total_people']}")
                    
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
                    # Initialize and run the crowd flow model
                    flow_model = CrowdFlowModel()
                    flow_results = flow_model.analyze(st.session_state.processed_data)
                    
                    # Store results in session state
                    st.session_state.flow_results = flow_results
                    
                    # Display results
                    st.success("Flow analysis complete!")
            
            if st.session_state.flow_results is not None:
                st.write("### Flow Results")
                
                flow_fig = create_flow_visualization(
                    st.session_state.processed_data,
                    st.session_state.flow_results["flow_vectors"]
                )
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
                metrics_fig = plot_crowd_metrics(
                    st.session_state.density_results,
                    st.session_state.flow_results
                )
                st.plotly_chart(metrics_fig, use_container_width=True)
                
                st.write("### Key Insights")
                st.write("- Areas with high density and low flow indicate potential congestion points")
                st.write("- Areas with high flow and moderate density indicate main thoroughfares")
                st.write("- Areas with low density and high flow indicate potential underutilized spaces")
            else:
                st.info("Complete both density and flow analysis to view combined metrics.")
        
        # Generate recommendations when both analyses are complete
        if st.session_state.density_results is not None and st.session_state.flow_results is not None:
            if st.button("Generate Recommendations"):
                st.session_state.recommendations = generate_recommendations(
                    st.session_state.density_results,
                    st.session_state.flow_results
                )
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
        st.write("### Report Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            event_name = st.text_input("Event Name", "Crowd Management Report")
            event_date = st.date_input("Event Date")
        
        with col2:
            include_visualizations = st.checkbox("Include Visualizations", value=True)
            include_recommendations = st.checkbox("Include Recommendations", value=True)
        
        if st.button("Generate Report"):
            with st.spinner("Generating comprehensive report..."):
                report_html = generate_report(
                    event_name=event_name,
                    event_date=event_date,
                    processed_data=st.session_state.processed_data,
                    density_results=st.session_state.density_results,
                    flow_results=st.session_state.flow_results,
                    recommendations=st.session_state.recommendations,
                    include_visualizations=include_visualizations,
                    include_recommendations=include_recommendations
                )
                
                # Display report preview
                st.write("### Report Preview")
                st.components.v1.html(report_html, height=600, scrolling=True)
                
                # Download button for the report
                st.download_button(
                    label="Download Report as HTML",
                    data=report_html,
                    file_name=f"{event_name.replace(' ', '_')}_report.html",
                    mime="text/html"
                )
