import os
import json
import psycopg2
from psycopg2.extras import Json
from datetime import datetime

class Database:
    """
    Database handler for the LiDAR Crowd Analytics application.
    Handles storage and retrieval of analysis results.
    """
    
    def __init__(self):
        """Initialize database connection from environment variables."""
        self.conn = None
        self.connect()
    
    def connect(self):
        """Connect to the PostgreSQL database."""
        try:
            # Connect using DATABASE_URL environment variable
            database_url = os.environ.get('DATABASE_URL')
            if database_url:
                self.conn = psycopg2.connect(database_url)
            else:
                # Or use individual connection parameters
                self.conn = psycopg2.connect(
                    host=os.environ.get('PGHOST'),
                    port=os.environ.get('PGPORT'),
                    user=os.environ.get('PGUSER'),
                    password=os.environ.get('PGPASSWORD'),
                    database=os.environ.get('PGDATABASE')
                )
            
            # Set autocommit mode
            self.conn.autocommit = True
            
            return True
        except Exception as e:
            print(f"Database connection error: {str(e)}")
            return False
    
    def create_event(self, name, event_date):
        """
        Create a new event in the database.
        
        Args:
            name (str): Event name
            event_date (date): Event date
            
        Returns:
            int: Event ID if created successfully, None otherwise
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO events (name, event_date) VALUES (%s, %s) RETURNING id",
                (name, event_date)
            )
            event_id = cursor.fetchone()[0]
            cursor.close()
            return event_id
        except Exception as e:
            print(f"Error creating event: {str(e)}")
            return None
    
    def create_analysis(self, event_id, analysis_type, processed_data):
        """
        Create a new analysis record in the database.
        
        Args:
            event_id (int): Event ID
            analysis_type (str): Type of analysis ('point_cloud', 'density', 'flow', etc.)
            processed_data (dict): Processed point cloud data
            
        Returns:
            int: Analysis ID if created successfully, None otherwise
        """
        try:
            dimensions = processed_data.get('dimensions', {})
            
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO analyses 
                (event_id, analysis_type, point_cloud_summary, total_points, width, length, height) 
                VALUES (%s, %s, %s, %s, %s, %s, %s) 
                RETURNING id
                """,
                (
                    event_id,
                    analysis_type,
                    Json({
                        'x_range': dimensions.get('x_range'),
                        'y_range': dimensions.get('y_range'),
                        'z_range': dimensions.get('z_range')
                    }),
                    len(processed_data.get('points', [])),
                    dimensions.get('width', 0),
                    dimensions.get('length', 0),
                    dimensions.get('height', 0)
                )
            )
            analysis_id = cursor.fetchone()[0]
            cursor.close()
            return analysis_id
        except Exception as e:
            print(f"Error creating analysis: {str(e)}")
            return None
    
    def save_density_results(self, analysis_id, density_results):
        """
        Save density analysis results to the database.
        
        Args:
            analysis_id (int): Analysis ID
            density_results (dict): Results from crowd density analysis
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO density_results 
                (analysis_id, total_people, avg_density, max_density, density_data, hotspots) 
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    analysis_id,
                    density_results.get('total_people', 0),
                    density_results.get('avg_density', 0.0),
                    density_results.get('max_density', 0.0),
                    Json({
                        'grid_shape': density_results.get('density_grid', []).shape if hasattr(density_results.get('density_grid', []), 'shape') else []
                    }),
                    Json(density_results.get('hotspots', []))
                )
            )
            cursor.close()
            return True
        except Exception as e:
            print(f"Error saving density results: {str(e)}")
            return False
    
    def save_flow_results(self, analysis_id, flow_results):
        """
        Save flow analysis results to the database.
        
        Args:
            analysis_id (int): Analysis ID
            flow_results (dict): Results from crowd flow analysis
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO flow_results 
                (analysis_id, avg_speed, dominant_direction, bottlenecks, flow_data) 
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    analysis_id,
                    flow_results.get('avg_speed', 0.0),
                    flow_results.get('dominant_direction', 'N/A'),
                    Json(flow_results.get('bottlenecks', [])),
                    Json({
                        'position_count': len(flow_results.get('flow_vectors', {}).get('positions', [])),
                        'avg_magnitude': float(flow_results.get('avg_speed', 0.0))
                    })
                )
            )
            cursor.close()
            return True
        except Exception as e:
            print(f"Error saving flow results: {str(e)}")
            return False
    
    def save_recommendations(self, analysis_id, recommendations):
        """
        Save crowd management recommendations to the database.
        
        Args:
            analysis_id (int): Analysis ID
            recommendations (dict): Crowd management recommendations
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO recommendations 
                (analysis_id, issues, actions, opportunities) 
                VALUES (%s, %s, %s, %s)
                """,
                (
                    analysis_id,
                    Json(recommendations.get('issues', [])),
                    Json(recommendations.get('actions', [])),
                    Json(recommendations.get('opportunities', []))
                )
            )
            cursor.close()
            return True
        except Exception as e:
            print(f"Error saving recommendations: {str(e)}")
            return False
    
    def save_report(self, analysis_id, report_name, report_html):
        """
        Save a generated report to the database.
        
        Args:
            analysis_id (int): Analysis ID
            report_name (str): Name of the report
            report_html (str): HTML content of the report
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO reports 
                (analysis_id, report_name, report_html) 
                VALUES (%s, %s, %s)
                """,
                (
                    analysis_id,
                    report_name,
                    report_html
                )
            )
            cursor.close()
            return True
        except Exception as e:
            print(f"Error saving report: {str(e)}")
            return False
    
    def get_all_events(self):
        """
        Get all events from the database.
        
        Returns:
            list: List of events (id, name, date)
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT id, name, event_date FROM events ORDER BY event_date DESC")
            events = cursor.fetchall()
            cursor.close()
            return events
        except Exception as e:
            print(f"Error getting events: {str(e)}")
            return []
    
    def get_analyses_for_event(self, event_id):
        """
        Get all analyses for a specific event.
        
        Args:
            event_id (int): Event ID
            
        Returns:
            list: List of analyses with basic info
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT a.id, a.analysis_type, a.analysis_date, a.total_points, 
                       d.total_people, d.avg_density, f.avg_speed
                FROM analyses a
                LEFT JOIN density_results d ON a.id = d.analysis_id
                LEFT JOIN flow_results f ON a.id = f.analysis_id
                WHERE a.event_id = %s
                ORDER BY a.analysis_date DESC
                """,
                (event_id,)
            )
            analyses = cursor.fetchall()
            cursor.close()
            return analyses
        except Exception as e:
            print(f"Error getting analyses: {str(e)}")
            return []
    
    def get_reports_for_event(self, event_id):
        """
        Get all reports for a specific event.
        
        Args:
            event_id (int): Event ID
            
        Returns:
            list: List of reports (id, name, date)
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT r.id, r.report_name, r.created_at
                FROM reports r
                JOIN analyses a ON r.analysis_id = a.id
                WHERE a.event_id = %s
                ORDER BY r.created_at DESC
                """,
                (event_id,)
            )
            reports = cursor.fetchall()
            cursor.close()
            return reports
        except Exception as e:
            print(f"Error getting reports: {str(e)}")
            return []
    
    def get_report_by_id(self, report_id):
        """
        Get a specific report by ID.
        
        Args:
            report_id (int): Report ID
            
        Returns:
            tuple: (report_name, report_html, created_at)
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT report_name, report_html, created_at
                FROM reports
                WHERE id = %s
                """,
                (report_id,)
            )
            report = cursor.fetchone()
            cursor.close()
            return report
        except Exception as e:
            print(f"Error getting report: {str(e)}")
            return None
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()