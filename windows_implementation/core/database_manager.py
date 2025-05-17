"""
Database Manager module for LiDAR Crowd Analytics
Handles database connection and operations
"""

import os
import sqlite3
import logging
import json
import numpy as np
import io
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages the SQLite database for LiDAR Crowd Analytics
    Can be extended to support PostgreSQL for enterprise deployments
    """
    
    def __init__(self, db_path='lidar_analytics.db'):
        """
        Initialize database connection
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.connected = self._connect()
        
        if self.connected:
            self._create_tables()
    
    def _connect(self):
        """Establish database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            # Enable foreign keys
            self.conn.execute("PRAGMA foreign_keys = ON")
            # Configure connection
            self.conn.row_factory = sqlite3.Row
            logger.info(f"Connected to database: {self.db_path}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {str(e)}")
            return False
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        try:
            cursor = self.conn.cursor()
            
            # Projects table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_date TEXT NOT NULL,
                modified_date TEXT NOT NULL,
                settings TEXT
            )
            ''')
            
            # Datasets table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                metadata TEXT,
                imported_date TEXT NOT NULL,
                point_count INTEGER NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
            )
            ''')
            
            # Dataset points table (binary storage)
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS dataset_points (
                dataset_id INTEGER PRIMARY KEY,
                points BLOB NOT NULL,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id) ON DELETE CASCADE
            )
            ''')
            
            # Analyses table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                dataset_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                results TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id) ON DELETE CASCADE
            )
            ''')
            
            # Reports table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                analysis_id INTEGER,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                content TEXT,
                file_path TEXT,
                created_date TEXT NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE,
                FOREIGN KEY (analysis_id) REFERENCES analyses (id) ON DELETE SET NULL
            )
            ''')
            
            self.conn.commit()
            logger.info("Database tables created or already exist")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error creating tables: {str(e)}")
            return False
    
    def insert_project(self, project_dict):
        """
        Insert a new project into the database
        
        Args:
            project_dict (dict): Project data
            
        Returns:
            int: Project ID if successful, None otherwise
        """
        if not self.conn:
            logger.error("No database connection")
            return None
            
        try:
            cursor = self.conn.cursor()
            
            # Convert settings to JSON if present
            settings_json = json.dumps(project_dict.get('settings', {}))
            
            cursor.execute('''
            INSERT INTO projects (name, created_date, modified_date, settings)
            VALUES (?, ?, ?, ?)
            ''', (
                project_dict['name'],
                project_dict['created_date'],
                project_dict['modified_date'],
                settings_json
            ))
            
            self.conn.commit()
            project_id = cursor.lastrowid
            logger.info(f"Inserted project: {project_dict['name']} (ID: {project_id})")
            return project_id
        except sqlite3.Error as e:
            logger.error(f"Error inserting project: {str(e)}")
            return None
    
    def update_project(self, project_id, project_dict):
        """
        Update an existing project in the database
        
        Args:
            project_id (int): Project ID
            project_dict (dict): Updated project data
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.conn:
            logger.error("No database connection")
            return False
            
        try:
            cursor = self.conn.cursor()
            
            # Convert settings to JSON if present
            settings_json = json.dumps(project_dict.get('settings', {}))
            
            cursor.execute('''
            UPDATE projects
            SET name = ?, modified_date = ?, settings = ?
            WHERE id = ?
            ''', (
                project_dict['name'],
                project_dict['modified_date'],
                settings_json,
                project_id
            ))
            
            self.conn.commit()
            logger.info(f"Updated project: {project_dict['name']} (ID: {project_id})")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error updating project: {str(e)}")
            return False
    
    def get_project(self, project_id):
        """
        Get a project by ID
        
        Args:
            project_id (int): Project ID
            
        Returns:
            dict: Project data if found, None otherwise
        """
        if not self.conn:
            logger.error("No database connection")
            return None
            
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            SELECT id, name, created_date, modified_date, settings
            FROM projects
            WHERE id = ?
            ''', (project_id,))
            
            row = cursor.fetchone()
            
            if row:
                # Convert row to dictionary
                project = dict(row)
                
                # Parse settings JSON
                if project['settings']:
                    project['settings'] = json.loads(project['settings'])
                else:
                    project['settings'] = {}
                
                return project
            else:
                logger.warning(f"Project not found: {project_id}")
                return None
        except sqlite3.Error as e:
            logger.error(f"Error getting project: {str(e)}")
            return None
    
    def get_all_projects(self):
        """
        Get all projects from the database
        
        Returns:
            list: List of project dictionaries
        """
        if not self.conn:
            logger.error("No database connection")
            return []
            
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            SELECT id, name, created_date, modified_date
            FROM projects
            ORDER BY modified_date DESC
            ''')
            
            rows = cursor.fetchall()
            
            # Convert rows to list of dictionaries
            projects = [dict(row) for row in rows]
            
            return projects
        except sqlite3.Error as e:
            logger.error(f"Error getting all projects: {str(e)}")
            return []
    
    def delete_project(self, project_id):
        """
        Delete a project from the database
        
        Args:
            project_id (int): Project ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.conn:
            logger.error("No database connection")
            return False
            
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            DELETE FROM projects
            WHERE id = ?
            ''', (project_id,))
            
            self.conn.commit()
            logger.info(f"Deleted project: {project_id}")
            return True
        except sqlite3.Error as e:
            logger.error(f"Error deleting project: {str(e)}")
            return False
    
    def insert_dataset(self, dataset_dict, points):
        """
        Insert a new dataset into the database
        
        Args:
            dataset_dict (dict): Dataset data
            points (numpy.ndarray): Point cloud data
            
        Returns:
            int: Dataset ID if successful, None otherwise
        """
        if not self.conn:
            logger.error("No database connection")
            return None
            
        try:
            cursor = self.conn.cursor()
            
            # Convert metadata to JSON if present
            metadata_json = json.dumps(dataset_dict.get('metadata', {}))
            
            # Insert dataset record
            cursor.execute('''
            INSERT INTO datasets (project_id, name, metadata, imported_date, point_count)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                dataset_dict['project_id'],
                dataset_dict['name'],
                metadata_json,
                dataset_dict['imported_date'],
                len(points)
            ))
            
            dataset_id = cursor.lastrowid
            
            # Save points to binary storage
            # Convert points to binary
            points_blob = io.BytesIO()
            np.save(points_blob, points)
            points_blob.seek(0)
            
            cursor.execute('''
            INSERT INTO dataset_points (dataset_id, points)
            VALUES (?, ?)
            ''', (dataset_id, points_blob.read()))
            
            self.conn.commit()
            logger.info(f"Inserted dataset: {dataset_dict['name']} (ID: {dataset_id})")
            return dataset_id
        except sqlite3.Error as e:
            logger.error(f"Error inserting dataset: {str(e)}")
            return None
    
    def get_dataset(self, dataset_id):
        """
        Get a dataset by ID (metadata only, not points)
        
        Args:
            dataset_id (int): Dataset ID
            
        Returns:
            dict: Dataset data if found, None otherwise
        """
        if not self.conn:
            logger.error("No database connection")
            return None
            
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            SELECT id, project_id, name, metadata, imported_date, point_count
            FROM datasets
            WHERE id = ?
            ''', (dataset_id,))
            
            row = cursor.fetchone()
            
            if row:
                # Convert row to dictionary
                dataset = dict(row)
                
                # Parse metadata JSON
                if dataset['metadata']:
                    dataset['metadata'] = json.loads(dataset['metadata'])
                else:
                    dataset['metadata'] = {}
                
                return dataset
            else:
                logger.warning(f"Dataset not found: {dataset_id}")
                return None
        except sqlite3.Error as e:
            logger.error(f"Error getting dataset: {str(e)}")
            return None
    
    def get_datasets_for_project(self, project_id):
        """
        Get all datasets for a project
        
        Args:
            project_id (int): Project ID
            
        Returns:
            list: List of dataset dictionaries
        """
        if not self.conn:
            logger.error("No database connection")
            return []
            
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            SELECT id, name, metadata, imported_date, point_count
            FROM datasets
            WHERE project_id = ?
            ORDER BY imported_date DESC
            ''', (project_id,))
            
            rows = cursor.fetchall()
            
            # Convert rows to list of dictionaries
            datasets = []
            for row in rows:
                dataset = dict(row)
                
                # Parse metadata JSON
                if dataset['metadata']:
                    dataset['metadata'] = json.loads(dataset['metadata'])
                else:
                    dataset['metadata'] = {}
                
                datasets.append(dataset)
            
            return datasets
        except sqlite3.Error as e:
            logger.error(f"Error getting datasets for project: {str(e)}")
            return []
    
    def get_dataset_points(self, dataset_id):
        """
        Get point cloud data for a dataset
        
        Args:
            dataset_id (int): Dataset ID
            
        Returns:
            numpy.ndarray: Point cloud data if found, None otherwise
        """
        if not self.conn:
            logger.error("No database connection")
            return None
            
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            SELECT points
            FROM dataset_points
            WHERE dataset_id = ?
            ''', (dataset_id,))
            
            row = cursor.fetchone()
            
            if row:
                # Load points from binary
                points_blob = io.BytesIO(row['points'])
                points = np.load(points_blob)
                return points
            else:
                logger.warning(f"Points not found for dataset: {dataset_id}")
                return None
        except sqlite3.Error as e:
            logger.error(f"Error getting dataset points: {str(e)}")
            return None
    
    def insert_analysis(self, project_id, dataset_id, results):
        """
        Insert analysis results into the database
        
        Args:
            project_id (int): Project ID
            dataset_id (int): Dataset ID
            results (dict): Analysis results
            
        Returns:
            int: Analysis ID if successful, None otherwise
        """
        if not self.conn:
            logger.error("No database connection")
            return None
            
        try:
            cursor = self.conn.cursor()
            
            # Convert results to JSON
            # Handle NumPy arrays
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                return obj
            
            results_json = json.dumps(results, default=convert_numpy)
            
            # Get timestamp from results or use current time
            timestamp = results.get('timestamp', datetime.now().isoformat())
            
            cursor.execute('''
            INSERT INTO analyses (project_id, dataset_id, timestamp, results)
            VALUES (?, ?, ?, ?)
            ''', (project_id, dataset_id, timestamp, results_json))
            
            self.conn.commit()
            analysis_id = cursor.lastrowid
            logger.info(f"Inserted analysis: {analysis_id}")
            return analysis_id
        except sqlite3.Error as e:
            logger.error(f"Error inserting analysis: {str(e)}")
            return None
    
    def get_analyses_for_project(self, project_id):
        """
        Get all analyses for a project
        
        Args:
            project_id (int): Project ID
            
        Returns:
            list: List of analysis dictionaries
        """
        if not self.conn:
            logger.error("No database connection")
            return []
            
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            SELECT id, dataset_id, timestamp, results
            FROM analyses
            WHERE project_id = ?
            ORDER BY timestamp DESC
            ''', (project_id,))
            
            rows = cursor.fetchall()
            
            # Convert rows to list of dictionaries
            analyses = []
            for row in rows:
                analysis = dict(row)
                
                # Parse results JSON
                if analysis['results']:
                    analysis['results'] = json.loads(analysis['results'])
                else:
                    analysis['results'] = {}
                
                analyses.append(analysis)
            
            return analyses
        except sqlite3.Error as e:
            logger.error(f"Error getting analyses for project: {str(e)}")
            return []
    
    def insert_report(self, project_id, analysis_id, name, report_type, content=None, file_path=None):
        """
        Insert a report into the database
        
        Args:
            project_id (int): Project ID
            analysis_id (int): Analysis ID
            name (str): Report name
            report_type (str): Report type (pdf, html, etc.)
            content (str, optional): Report content for HTML reports
            file_path (str, optional): Path to report file
            
        Returns:
            int: Report ID if successful, None otherwise
        """
        if not self.conn:
            logger.error("No database connection")
            return None
            
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            INSERT INTO reports (project_id, analysis_id, name, type, content, file_path, created_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                project_id, 
                analysis_id, 
                name, 
                report_type, 
                content, 
                file_path, 
                datetime.now().isoformat()
            ))
            
            self.conn.commit()
            report_id = cursor.lastrowid
            logger.info(f"Inserted report: {name} (ID: {report_id})")
            return report_id
        except sqlite3.Error as e:
            logger.error(f"Error inserting report: {str(e)}")
            return None
    
    def get_reports_for_project(self, project_id):
        """
        Get all reports for a project
        
        Args:
            project_id (int): Project ID
            
        Returns:
            list: List of report dictionaries
        """
        if not self.conn:
            logger.error("No database connection")
            return []
            
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
            SELECT id, analysis_id, name, type, file_path, created_date
            FROM reports
            WHERE project_id = ?
            ORDER BY created_date DESC
            ''', (project_id,))
            
            rows = cursor.fetchall()
            
            # Convert rows to list of dictionaries
            reports = [dict(row) for row in rows]
            
            return reports
        except sqlite3.Error as e:
            logger.error(f"Error getting reports for project: {str(e)}")
            return []
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
            self.conn = None