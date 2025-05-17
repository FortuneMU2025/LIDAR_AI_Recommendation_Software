"""
Project Manager module for LiDAR Crowd Analytics
Handles project creation, saving, loading, and management
"""

import os
import json
import logging
import numpy as np
import pickle
from datetime import datetime
from .database_manager import DatabaseManager

logger = logging.getLogger(__name__)

class Project:
    """Represents a LiDAR Crowd Analytics project"""
    
    def __init__(self, name):
        self.id = None  # Set when saved to database
        self.name = name
        self.created_date = datetime.now()
        self.modified_date = self.created_date
        self.datasets = {}  # Dictionary of dataset objects
        self.analyses = {}  # Dictionary of analysis results
        self.reports = {}   # Dictionary of generated reports
        self.settings = {}  # Project settings
        self.modified = False
        
    def to_dict(self):
        """Convert project to dictionary for serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'created_date': self.created_date.isoformat(),
            'modified_date': self.modified_date.isoformat(),
            'settings': self.settings
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create a project from dictionary data"""
        project = cls(data['name'])
        project.id = data.get('id')
        project.created_date = datetime.fromisoformat(data['created_date'])
        project.modified_date = datetime.fromisoformat(data['modified_date'])
        project.settings = data.get('settings', {})
        return project


class Dataset:
    """Represents a LiDAR dataset within a project"""
    
    def __init__(self, name, points, metadata=None):
        self.id = None  # Set when saved to database
        self.name = name
        self.points = points  # NumPy array of points
        self.metadata = metadata or {}
        self.imported_date = datetime.now()
        
    def to_dict(self):
        """Convert dataset to dictionary for serialization"""
        # Note: points are not included as they're stored separately
        return {
            'id': self.id,
            'name': self.name,
            'metadata': self.metadata,
            'imported_date': self.imported_date.isoformat(),
            'point_count': len(self.points)
        }


class ProjectManager:
    """Manages LiDAR Crowd Analytics projects"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.current_project = None
        self.current_dataset = None
        
    def create_project(self, name):
        """Create a new project"""
        try:
            project = Project(name)
            
            # Save to database to get ID
            project_dict = project.to_dict()
            project_id = self.db_manager.insert_project(project_dict)
            
            if project_id:
                project.id = project_id
                self.current_project = project
                logger.info(f"Created new project: {name} (ID: {project_id})")
                return True
            else:
                logger.error(f"Failed to create project in database: {name}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating project: {str(e)}")
            return False
    
    def open_project(self, file_path):
        """
        Open an existing project file (.lcap)
        For standalone file-based operation
        """
        try:
            with open(file_path, 'rb') as f:
                # Load project data
                project_data = pickle.load(f)
                
                # Create project object
                project = Project.from_dict(project_data['project'])
                
                # Load datasets if they exist
                if 'datasets' in project_data:
                    for dataset_dict in project_data['datasets']:
                        dataset_points = project_data['dataset_points'].get(dataset_dict['name'])
                        if dataset_points is not None:
                            dataset = Dataset(dataset_dict['name'], dataset_points, dataset_dict.get('metadata'))
                            project.datasets[dataset.name] = dataset
                
                # Set as current project
                self.current_project = project
                logger.info(f"Opened project: {project.name} from {file_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error opening project file: {str(e)}")
            return False
    
    def save_project(self, file_path):
        """
        Save current project to file (.lcap)
        For standalone file-based operation
        """
        if not self.current_project:
            logger.error("No project to save")
            return False
            
        try:
            # Update modified date
            self.current_project.modified_date = datetime.now()
            self.current_project.modified = False
            
            # Prepare data for saving
            project_data = {
                'project': self.current_project.to_dict(),
                'datasets': [],
                'dataset_points': {}
            }
            
            # Add datasets
            for dataset in self.current_project.datasets.values():
                project_data['datasets'].append(dataset.to_dict())
                project_data['dataset_points'][dataset.name] = dataset.points
            
            # Save to file
            with open(file_path, 'wb') as f:
                pickle.dump(project_data, f)
            
            logger.info(f"Saved project to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving project: {str(e)}")
            return False
    
    def load_project_from_db(self, project_id):
        """Load a project from the database"""
        try:
            # Get project data
            project_data = self.db_manager.get_project(project_id)
            
            if project_data:
                # Create project object
                project = Project.from_dict(project_data)
                
                # Load datasets
                datasets = self.db_manager.get_datasets_for_project(project_id)
                for dataset_data in datasets:
                    # Load points
                    points_data = self.db_manager.get_dataset_points(dataset_data['id'])
                    if points_data:
                        dataset = Dataset(dataset_data['name'], points_data, dataset_data.get('metadata'))
                        dataset.id = dataset_data['id']
                        project.datasets[dataset.name] = dataset
                
                # Set as current project
                self.current_project = project
                logger.info(f"Loaded project from database: {project.name} (ID: {project_id})")
                return True
            else:
                logger.error(f"Project not found in database: {project_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading project from database: {str(e)}")
            return False
    
    def get_all_projects(self):
        """Get a list of all projects in the database"""
        try:
            return self.db_manager.get_all_projects()
        except Exception as e:
            logger.error(f"Error getting projects: {str(e)}")
            return []
    
    def get_project(self, project_id):
        """Get a specific project by ID"""
        try:
            project_data = self.db_manager.get_project(project_id)
            if project_data:
                return Project.from_dict(project_data)
            return None
        except Exception as e:
            logger.error(f"Error getting project {project_id}: {str(e)}")
            return None
    
    def add_dataset(self, name, dataset):
        """Add a dataset to the current project"""
        if not self.current_project:
            logger.error("No project is open to add dataset")
            return False
            
        try:
            # Add to project
            self.current_project.datasets[name] = dataset
            self.current_project.modified = True
            
            # Add to database if project has an ID
            if self.current_project.id:
                dataset_dict = dataset.to_dict()
                dataset_dict['project_id'] = self.current_project.id
                dataset_id = self.db_manager.insert_dataset(dataset_dict, dataset.points)
                
                if dataset_id:
                    dataset.id = dataset_id
                    logger.info(f"Added dataset to database: {name} (ID: {dataset_id})")
                else:
                    logger.warning(f"Added dataset to project but failed to save to database: {name}")
            
            logger.info(f"Added dataset to project: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding dataset: {str(e)}")
            return False
    
    def get_dataset(self, dataset_id):
        """Get a specific dataset by ID"""
        try:
            # Check if it's in the current project
            for dataset in self.current_project.datasets.values():
                if dataset.id == dataset_id:
                    return dataset
            
            # If not found in memory, try to load from database
            dataset_data = self.db_manager.get_dataset(dataset_id)
            if dataset_data:
                points_data = self.db_manager.get_dataset_points(dataset_id)
                if points_data:
                    dataset = Dataset(dataset_data['name'], points_data, dataset_data.get('metadata'))
                    dataset.id = dataset_id
                    return dataset
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting dataset {dataset_id}: {str(e)}")
            return None
    
    def run_analysis(self, parameters):
        """
        Run crowd analysis on the current dataset
        
        Args:
            parameters (dict): Analysis parameters
            
        Returns:
            dict: Analysis results
        """
        if not self.current_project or not self.current_dataset:
            logger.error("No project or dataset selected for analysis")
            return None
            
        try:
            logger.info(f"Running analysis with parameters: {parameters}")
            
            # This is where the actual analysis would occur
            # For now, we'll create simulated results for demonstration purposes
            
            # Extract points from dataset
            points = self.current_dataset.points
            
            # Simulated density analysis
            total_people = max(10, int(len(points) / 1000))  # Simulate people count
            avg_density = total_people / 100.0  # People per square meter
            max_density = avg_density * 2.5  # Some hotspots have higher density
            
            # Create simulated density map
            x_min, y_min = np.min(points[:, :2], axis=0)
            x_max, y_max = np.max(points[:, :2], axis=0)
            grid_size = 1.0
            x_grid = np.arange(x_min, x_max + grid_size, grid_size)
            y_grid = np.arange(y_min, y_max + grid_size, grid_size)
            density_grid = np.zeros((len(y_grid)-1, len(x_grid)-1))
            
            # Add some simulated density patterns
            for i in range(len(x_grid)-1):
                for j in range(len(y_grid)-1):
                    x = (x_grid[i] + x_grid[i+1]) / 2
                    y = (y_grid[j] + y_grid[j+1]) / 2
                    # Create some hotspots
                    dist_to_center = np.sqrt((x - (x_min + x_max)/2)**2 + (y - (y_min + y_max)/2)**2)
                    dist_to_corner = np.sqrt((x - x_max)**2 + (y - y_max)**2)
                    density_grid[j, i] = 2.0 * np.exp(-dist_to_center/10.0) + 1.5 * np.exp(-dist_to_corner/15.0)
            
            # Identify hotspots
            hotspot_threshold = 1.5  # People per square meter
            hotspot_locations = []
            
            for j in range(density_grid.shape[0]):
                for i in range(density_grid.shape[1]):
                    if density_grid[j, i] >= hotspot_threshold:
                        x = (x_grid[i] + x_grid[i+1]) / 2
                        y = (y_grid[j] + y_grid[j+1]) / 2
                        density = density_grid[j, i]
                        hotspot_locations.append({
                            'x': float(x),
                            'y': float(y),
                            'density': float(density)
                        })
            
            # Sort hotspots by density
            hotspot_locations = sorted(hotspot_locations, key=lambda x: x['density'], reverse=True)[:5]
            
            # Simulated flow analysis
            avg_speed = 1.2  # m/s
            dominant_direction = "NE"
            
            # Simulated bottlenecks
            bottlenecks = [
                {'x': float(x_min + (x_max - x_min) * 0.7), 'y': float(y_min + (y_max - y_min) * 0.3), 'severity': 8},
                {'x': float(x_min + (x_max - x_min) * 0.4), 'y': float(y_min + (y_max - y_min) * 0.6), 'severity': 6},
                {'x': float(x_min + (x_max - x_min) * 0.2), 'y': float(y_min + (y_max - y_min) * 0.8), 'severity': 4}
            ]
            
            # Create results dictionary
            results = {
                'total_people': total_people,
                'avg_density': avg_density,
                'max_density': max_density,
                'density_map': density_grid,
                'hotspots': hotspot_locations,
                'avg_speed': avg_speed,
                'dominant_direction': dominant_direction,
                'bottlenecks': bottlenecks,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store results in project
            analysis_id = len(self.current_project.analyses) + 1
            self.current_project.analyses[analysis_id] = results
            self.current_project.modified = True
            
            # Save to database if project has an ID
            if self.current_project.id:
                self.db_manager.insert_analysis(self.current_project.id, self.current_dataset.id, results)
            
            logger.info(f"Analysis completed with {total_people} people detected")
            return results
            
        except Exception as e:
            logger.error(f"Error running analysis: {str(e)}")
            return None
    
    def generate_pdf_report(self, file_path):
        """
        Generate a PDF report of the analysis results
        
        Args:
            file_path (str): Path to save the PDF report
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.current_project or not self.current_project.analyses:
            logger.error("No analysis results to generate report")
            return False
            
        try:
            # In a real implementation, this would use a PDF generation library
            # For now, we'll just simulate PDF creation
            
            logger.info(f"Generated PDF report: {file_path}")
            
            # For simulation purposes, create a simple text file with PDF extension
            with open(file_path, 'w') as f:
                f.write("LiDAR Crowd Analytics PDF Report\n")
                f.write("===============================\n\n")
                f.write(f"Project: {self.current_project.name}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for analysis_id, results in self.current_project.analyses.items():
                    f.write(f"Analysis #{analysis_id}\n")
                    f.write(f"Total People: {results['total_people']}\n")
                    f.write(f"Average Density: {results['avg_density']:.2f} people/m²\n")
                    f.write(f"Maximum Density: {results['max_density']:.2f} people/m²\n")
                    f.write(f"Average Speed: {results['avg_speed']:.2f} m/s\n")
                    f.write(f"Dominant Direction: {results['dominant_direction']}\n\n")
                    
                    f.write("Hotspots:\n")
                    for hotspot in results['hotspots']:
                        f.write(f"  - Location: ({hotspot['x']:.1f}, {hotspot['y']:.1f}), Density: {hotspot['density']:.2f} people/m²\n")
                    
                    f.write("\nBottlenecks:\n")
                    for bottleneck in results['bottlenecks']:
                        f.write(f"  - Location: ({bottleneck['x']:.1f}, {bottleneck['y']:.1f}), Severity: {bottleneck['severity']}/10\n")
            
            # Save report reference in project
            report_id = len(self.current_project.reports) + 1
            self.current_project.reports[report_id] = {
                'file_path': file_path,
                'type': 'pdf',
                'timestamp': datetime.now().isoformat()
            }
            self.current_project.modified = True
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            return False
    
    def generate_html_report(self, file_path):
        """
        Generate an HTML report of the analysis results
        
        Args:
            file_path (str): Path to save the HTML report
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.current_project or not self.current_project.analyses:
            logger.error("No analysis results to generate report")
            return False
            
        try:
            # In a real implementation, this would use proper HTML templates
            # For now, we'll generate a simple HTML report
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>LiDAR Crowd Analytics Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .hotspot {{ color: #e74c3c; }}
                    .bottleneck {{ color: #e67e22; }}
                </style>
            </head>
            <body>
                <h1>LiDAR Crowd Analytics Report</h1>
                <p><strong>Project:</strong> {self.current_project.name}</p>
                <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            """
            
            for analysis_id, results in self.current_project.analyses.items():
                html += f"""
                <h2>Analysis #{analysis_id}</h2>
                <h3>Summary</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total People</td><td>{results['total_people']}</td></tr>
                    <tr><td>Average Density</td><td>{results['avg_density']:.2f} people/m²</td></tr>
                    <tr><td>Maximum Density</td><td>{results['max_density']:.2f} people/m²</td></tr>
                    <tr><td>Average Speed</td><td>{results['avg_speed']:.2f} m/s</td></tr>
                    <tr><td>Dominant Direction</td><td>{results['dominant_direction']}</td></tr>
                </table>
                
                <h3>Hotspots</h3>
                <table>
                    <tr><th>Location (X, Y)</th><th>Density (people/m²)</th></tr>
                """
                
                for hotspot in results['hotspots']:
                    html += f"""
                    <tr class="hotspot">
                        <td>({hotspot['x']:.1f}, {hotspot['y']:.1f})</td>
                        <td>{hotspot['density']:.2f}</td>
                    </tr>
                    """
                
                html += """
                </table>
                
                <h3>Bottlenecks</h3>
                <table>
                    <tr><th>Location (X, Y)</th><th>Severity (1-10)</th></tr>
                """
                
                for bottleneck in results['bottlenecks']:
                    html += f"""
                    <tr class="bottleneck">
                        <td>({bottleneck['x']:.1f}, {bottleneck['y']:.1f})</td>
                        <td>{bottleneck['severity']}/10</td>
                    </tr>
                    """
                
                html += """
                </table>
                """
            
            html += """
            </body>
            </html>
            """
            
            with open(file_path, 'w') as f:
                f.write(html)
            
            logger.info(f"Generated HTML report: {file_path}")
            
            # Save report reference in project
            report_id = len(self.current_project.reports) + 1
            self.current_project.reports[report_id] = {
                'file_path': file_path,
                'type': 'html',
                'timestamp': datetime.now().isoformat()
            }
            self.current_project.modified = True
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            return False
    
    def export_csv_data(self, file_path):
        """Export analysis results as CSV"""
        if not self.current_project or not self.current_project.analyses:
            logger.error("No analysis results to export")
            return False
            
        try:
            # In a real implementation, this would use pandas or csv module
            # For now, we'll generate a simple CSV file
            
            with open(file_path, 'w') as f:
                # Write header
                f.write("analysis_id,metric,value\n")
                
                for analysis_id, results in self.current_project.analyses.items():
                    f.write(f"{analysis_id},total_people,{results['total_people']}\n")
                    f.write(f"{analysis_id},avg_density,{results['avg_density']}\n")
                    f.write(f"{analysis_id},max_density,{results['max_density']}\n")
                    f.write(f"{analysis_id},avg_speed,{results['avg_speed']}\n")
                    f.write(f"{analysis_id},dominant_direction,{results['dominant_direction']}\n")
            
            logger.info(f"Exported CSV data: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting CSV data: {str(e)}")
            return False
    
    def export_json_data(self, file_path):
        """Export analysis results as JSON"""
        if not self.current_project or not self.current_project.analyses:
            logger.error("No analysis results to export")
            return False
            
        try:
            # Export analysis results as JSON
            
            # Convert NumPy arrays to lists for JSON serialization
            export_data = {}
            
            for analysis_id, results in self.current_project.analyses.items():
                analysis_data = results.copy()
                
                # Convert NumPy arrays to lists
                if 'density_map' in analysis_data:
                    density_map = analysis_data['density_map']
                    if hasattr(density_map, 'tolist'):
                        analysis_data['density_map'] = density_map.tolist()
                
                export_data[str(analysis_id)] = analysis_data
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported JSON data: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting JSON data: {str(e)}")
            return False