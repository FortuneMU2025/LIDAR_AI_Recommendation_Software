import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from datetime import datetime
from utils.visualization import create_density_heatmap, create_flow_visualization, plot_crowd_metrics

def generate_report(event_name, event_date, processed_data, density_results, flow_results, 
                   recommendations, include_visualizations=True, include_recommendations=True):
    """
    Generate a comprehensive HTML report from the analysis results.
    
    Args:
        event_name (str): Name of the event
        event_date (datetime.date): Date of the event
        processed_data (dict): Processed point cloud data
        density_results (dict): Results from crowd density analysis
        flow_results (dict): Results from crowd flow analysis
        recommendations (dict): Crowd management recommendations
        include_visualizations (bool): Whether to include visualizations in the report
        include_recommendations (bool): Whether to include recommendations in the report
        
    Returns:
        str: HTML report as a string
    """
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
            h1 {{
                color: #2C3E50;
            }}
            h2 {{
                color: #3498DB;
                margin-top: 30px;
            }}
            h3 {{
                color: #2980B9;
            }}
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
            .metric-description {{
                font-size: 14px;
                color: #7f8c8d;
                margin-top: 5px;
            }}
            .image-container {{
                margin: 20px 0;
                text-align: center;
            }}
            .image-container img {{
                max-width: 100%;
                border-radius: 5px;
                box-shadow: 0 3px 6px rgba(0,0,0,0.1);
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
            .recommendation-title {{
                font-weight: bold;
                margin-bottom: 5px;
            }}
            .recommendation-priority {{
                display: inline-block;
                font-size: 12px;
                padding: 3px 8px;
                border-radius: 3px;
                color: white;
                margin-left: 10px;
            }}
            .priority-high {{
                background-color: #e74c3c;
            }}
            .priority-medium {{
                background-color: #f39c12;
            }}
            .priority-low {{
                background-color: #2ecc71;
            }}
            .footer {{
                text-align: center;
                margin-top: 50px;
                color: #7f8c8d;
                font-size: 14px;
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
            th {{
                background-color: #f8f9fa;
                font-weight: bold;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Crowd Analysis Report</h1>
            <h2>{event_name}</h2>
            <p>Date: {event_date.strftime('%B %d, %Y') if event_date else 'N/A'}</p>
            <p>Report generated: {datetime.now().strftime('%B %d, %Y %H:%M')}</p>
        </div>
        
        <div class="section">
            <h2>Executive Summary</h2>
            <p>This report provides a comprehensive analysis of crowd density and flow patterns based on LiDAR data. It includes metrics, visualizations, and actionable recommendations for crowd management.</p>
            
            <div class="metric-container">
                <div class="metric-box">
                    <div class="metric-title">Total People</div>
                    <div class="metric-value">{density_results['total_people']}</div>
                    <div class="metric-description">Estimated number of people detected in the dataset</div>
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
    
    if include_visualizations:
        # Generate density heatmap image
        density_fig = create_density_heatmap(
            processed_data,
            density_data=density_results["density_map"],
            projection_dims=["x", "y"],
            resolution=100,
            as_heatmap=True
        )
        
        # Convert plotly figure to base64 image
        density_img = fig_to_base64(density_fig)
        
        html_report += f"""
            <div class="image-container">
                <img src="data:image/png;base64,{density_img}" alt="Crowd Density Heatmap">
                <p><em>Figure 1: Crowd density heatmap showing people per square meter.</em></p>
            </div>
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
                risk_class = "low"
            elif hotspot['density'] < 2.5:
                risk_level = "Moderate"
                risk_class = "medium"
            elif hotspot['density'] < 4.0:
                risk_level = "High"
                risk_class = "high"
            else:
                risk_level = "Critical"
                risk_class = "high"
            
            html_report += f"""
                <tr>
                    <td>({hotspot['x']:.1f}, {hotspot['y']:.1f})</td>
                    <td>{hotspot['density']:.2f}</td>
                    <td class="priority-{risk_class}">{risk_level}</td>
                </tr>
            """
        
        html_report += """
            </table>
        """
    
    html_report += """
        </div>
    """
    
    # Add flow analysis section
    html_report += """
        <div class="section">
            <h2>Crowd Flow Analysis</h2>
            <p>Analysis of crowd movement patterns and flow dynamics.</p>
    """
    
    if include_visualizations:
        # Generate flow visualization image
        flow_fig = create_flow_visualization(
            processed_data,
            flow_results["flow_vectors"]
        )
        
        # Convert plotly figure to base64 image
        flow_img = fig_to_base64(flow_fig)
        
        html_report += f"""
            <div class="image-container">
                <img src="data:image/png;base64,{flow_img}" alt="Crowd Flow Visualization">
                <p><em>Figure 2: Crowd flow visualization showing movement patterns and speed.</em></p>
            </div>
        """
    
    # Add flow statistics
    html_report += f"""
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
                priority_class = "low"
            elif bottleneck['severity'] <= 6:
                priority = "Medium"
                priority_class = "medium"
            else:
                priority = "High"
                priority_class = "high"
            
            html_report += f"""
                <tr>
                    <td>({bottleneck['x']:.1f}, {bottleneck['y']:.1f})</td>
                    <td>{bottleneck['severity']}</td>
                    <td class="priority-{priority_class}">{priority}</td>
                </tr>
            """
        
        html_report += """
            </table>
        """
    
    html_report += """
        </div>
    """
    
    # Add combined metrics section if visualizations are included
    if include_visualizations:
        html_report += """
            <div class="section">
                <h2>Combined Analysis</h2>
                <p>Integrated analysis of crowd density and flow patterns.</p>
        """
        
        # Generate combined metrics visualization
        combined_fig = plot_crowd_metrics(
            density_results,
            flow_results
        )
        
        # Convert plotly figure to base64 image
        combined_img = fig_to_base64(combined_fig)
        
        html_report += f"""
            <div class="image-container">
                <img src="data:image/png;base64,{combined_img}" alt="Combined Crowd Metrics">
                <p><em>Figure 3: Combined analysis showing relationship between crowd density and flow.</em></p>
            </div>
            
            <h3>Key Insights</h3>
            <ul>
                <li>Areas with high density and low flow indicate potential congestion points</li>
                <li>Areas with high flow and moderate density indicate main thoroughfares</li>
                <li>Areas with low density and high flow indicate potential underutilized spaces</li>
            </ul>
        </div>
        """
    
    # Add recommendations section
    if include_recommendations and recommendations:
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
                    <div class="recommendation-title">
                        {action['title']}
                        <span class="recommendation-priority priority-{priority_class}">{action['priority']} Priority</span>
                    </div>
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
                    <div class="recommendation-title">{opportunity['title']}</div>
                    <p><strong>Potential impact:</strong> {opportunity['impact']}</p>
                    <p>{opportunity['description']}</p>
                </div>
            """
        
        html_report += """
            </div>
        """
    
    # Add footer
    html_report += f"""
        <div class="footer">
            <p>Report generated by 3D LiDAR Crowd Analytics System</p>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </body>
    </html>
    """
    
    return html_report

def fig_to_base64(fig):
    """
    Convert a plotly figure to a base64 encoded string.
    
    Args:
        fig (plotly.graph_objects.Figure): Plotly figure to convert
        
    Returns:
        str: Base64 encoded string of the image
    """
    buf = BytesIO()
    fig.write_image(buf, format="png", width=800, height=500, scale=2)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    return img_str
