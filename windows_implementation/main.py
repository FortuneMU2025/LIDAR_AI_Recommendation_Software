"""
LiDAR Crowd Analytics - Windows Application
Main application entry point
"""

import sys
import os
import logging
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QDockWidget, QAction, 
                            QFileDialog, QMessageBox, QVBoxLayout, QWidget,
                            QLabel, QPushButton, QTabWidget, QSplitter, QToolBar,
                            QStatusBar, QMenu, QTreeView, QListWidget, QTextEdit)
from PyQt5.QtCore import Qt, QSize, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QStandardItemModel, QStandardItem

# Import local modules
from gui.visualization_widget import VisualizationWidget
from gui.ribbon_menu import RibbonMenu
from gui.project_explorer import ProjectExplorer
from gui.analysis_panel import AnalysisPanel
from gui.results_panel import ResultsPanel
from core.project_manager import ProjectManager
from core.data_loader import DataLoader
from core.database_manager import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        
        # Initialize core components
        self.project_manager = ProjectManager()
        self.data_loader = DataLoader()
        self.db_manager = DatabaseManager()
        
        # Set window properties
        self.setWindowTitle("LiDAR Crowd Analytics")
        self.setMinimumSize(1200, 800)
        
        # Initialize UI
        self._create_ui()
        self._create_menu()
        self._create_statusbar()
        self._create_connections()
        
        # Show welcome message
        self.statusBar().showMessage("Ready. Create a new project or open an existing one to get started.")
        
        logger.info("Application initialized successfully")
    
    def _create_ui(self):
        """Create the main UI components"""
        # Central widget with splitter
        self.central_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(self.central_splitter)
        
        # Left panel for project explorer
        self.project_explorer = ProjectExplorer(self.project_manager)
        self.central_splitter.addWidget(self.project_explorer)
        
        # Right area with visualization and results
        self.right_area = QSplitter(Qt.Vertical)
        
        # 3D visualization widget
        self.visualization_widget = VisualizationWidget()
        self.right_area.addWidget(self.visualization_widget)
        
        # Results panel at bottom
        self.results_panel = ResultsPanel()
        self.right_area.addWidget(self.results_panel)
        
        # Add right area to main splitter
        self.central_splitter.addWidget(self.right_area)
        
        # Set initial splitter sizes
        self.central_splitter.setSizes([250, 950])
        self.right_area.setSizes([600, 200])
        
        # Analysis panel as dock widget
        self.analysis_panel = AnalysisPanel()
        self.analysis_dock = QDockWidget("Analysis", self)
        self.analysis_dock.setAllowedAreas(Qt.RightDockWidgetArea)
        self.analysis_dock.setWidget(self.analysis_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.analysis_dock)
        
        # Ribbon menu (custom widget)
        self.ribbon = RibbonMenu()
        self.addToolBar(Qt.TopToolBarArea, self.ribbon)
    
    def _create_menu(self):
        """Create the main menu"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        new_project_action = QAction(QIcon("icons/new.png"), "&New Project", self)
        new_project_action.setShortcut("Ctrl+N")
        new_project_action.setStatusTip("Create a new project")
        new_project_action.triggered.connect(self._on_new_project)
        file_menu.addAction(new_project_action)
        
        open_project_action = QAction(QIcon("icons/open.png"), "&Open Project", self)
        open_project_action.setShortcut("Ctrl+O")
        open_project_action.setStatusTip("Open an existing project")
        open_project_action.triggered.connect(self._on_open_project)
        file_menu.addAction(open_project_action)
        
        save_project_action = QAction(QIcon("icons/save.png"), "&Save Project", self)
        save_project_action.setShortcut("Ctrl+S")
        save_project_action.setStatusTip("Save the current project")
        save_project_action.triggered.connect(self._on_save_project)
        file_menu.addAction(save_project_action)
        
        file_menu.addSeparator()
        
        import_action = QAction(QIcon("icons/import.png"), "&Import Data", self)
        import_action.setStatusTip("Import LiDAR data into the project")
        import_action.triggered.connect(self._on_import_data)
        file_menu.addAction(import_action)
        
        export_action = QAction(QIcon("icons/export.png"), "&Export Results", self)
        export_action.setStatusTip("Export analysis results")
        export_action.triggered.connect(self._on_export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction(QIcon("icons/exit.png"), "E&xit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu("&Analysis")
        
        run_analysis_action = QAction(QIcon("icons/analyze.png"), "&Run Analysis", self)
        run_analysis_action.setShortcut("F5")
        run_analysis_action.setStatusTip("Run crowd analysis on loaded data")
        run_analysis_action.triggered.connect(self._on_run_analysis)
        analysis_menu.addAction(run_analysis_action)
        
        generate_report_action = QAction(QIcon("icons/report.png"), "&Generate Report", self)
        generate_report_action.setStatusTip("Generate a comprehensive report")
        generate_report_action.triggered.connect(self._on_generate_report)
        analysis_menu.addAction(generate_report_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        view_menu.addAction(self.analysis_dock.toggleViewAction())
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction(QIcon("icons/about.png"), "&About", self)
        about_action.setStatusTip("About LiDAR Crowd Analytics")
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)
    
    def _create_statusbar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add permanent widgets
        self.status_project = QLabel("No project open")
        self.status_bar.addPermanentWidget(self.status_project)
        
        self.status_points = QLabel("Points: 0")
        self.status_bar.addPermanentWidget(self.status_points)
    
    def _create_connections(self):
        """Connect signals and slots"""
        # Connect project explorer signals
        self.project_explorer.project_selected.connect(self._on_project_selected)
        self.project_explorer.dataset_selected.connect(self._on_dataset_selected)
        
        # Connect analysis panel signals
        self.analysis_panel.analysis_requested.connect(self._on_run_analysis)
        
        # Connect ribbon menu signals
        self.ribbon.import_clicked.connect(self._on_import_data)
        self.ribbon.analysis_clicked.connect(self._on_run_analysis)
        self.ribbon.visualization_changed.connect(self.visualization_widget.change_visualization_mode)
    
    def _on_new_project(self):
        """Handle new project creation"""
        # Get project details
        project_name = "New Project"  # In real app, would show a dialog
        
        # Create the project
        success = self.project_manager.create_project(project_name)
        
        if success:
            self.project_explorer.refresh()
            self.status_project.setText(f"Project: {project_name}")
            self.statusBar().showMessage(f"Created new project: {project_name}")
        else:
            QMessageBox.warning(self, "Error", "Failed to create new project.")
    
    def _on_open_project(self):
        """Handle opening an existing project"""
        # Get project file
        project_file, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "LiDAR Crowd Analytics Projects (*.lcap);;All Files (*)"
        )
        
        if project_file:
            success = self.project_manager.open_project(project_file)
            
            if success:
                project_name = self.project_manager.current_project.name
                self.project_explorer.refresh()
                self.status_project.setText(f"Project: {project_name}")
                self.statusBar().showMessage(f"Opened project: {project_name}")
            else:
                QMessageBox.warning(self, "Error", "Failed to open project.")
    
    def _on_save_project(self):
        """Handle saving the current project"""
        if not self.project_manager.current_project:
            QMessageBox.information(self, "No Project", "No project is currently open.")
            return
        
        # Get save location
        project_file, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "", "LiDAR Crowd Analytics Projects (*.lcap);;All Files (*)"
        )
        
        if project_file:
            success = self.project_manager.save_project(project_file)
            
            if success:
                self.statusBar().showMessage(f"Project saved to {project_file}")
            else:
                QMessageBox.warning(self, "Error", "Failed to save project.")
    
    def _on_import_data(self):
        """Handle importing LiDAR data"""
        if not self.project_manager.current_project:
            QMessageBox.information(self, "No Project", "Please create or open a project first.")
            return
        
        # Get data file
        data_files, _ = QFileDialog.getOpenFileNames(
            self, "Import LiDAR Data", "", 
            "All Supported Files (*.las *.laz *.pcd *.ply *.xyz *.csv);;LAS Files (*.las);;LAZ Files (*.laz);;PCD Files (*.pcd);;PLY Files (*.ply);;XYZ Files (*.xyz);;CSV Files (*.csv);;All Files (*)"
        )
        
        if data_files:
            for file_path in data_files:
                try:
                    # Load the data
                    dataset = self.data_loader.load_file(file_path)
                    
                    # Add to project
                    dataset_name = os.path.basename(file_path)
                    self.project_manager.add_dataset(dataset_name, dataset)
                    
                    # Update UI
                    self.project_explorer.refresh()
                    self.status_points.setText(f"Points: {len(dataset.points):,}")
                    self.statusBar().showMessage(f"Imported {dataset_name} with {len(dataset.points):,} points")
                    
                    # Visualize the data
                    self.visualization_widget.display_point_cloud(dataset.points)
                except Exception as e:
                    QMessageBox.warning(self, "Import Error", f"Failed to import {file_path}: {str(e)}")
    
    def _on_export_results(self):
        """Handle exporting analysis results"""
        if not self.project_manager.current_project or not hasattr(self.project_manager.current_project, 'results') or not self.project_manager.current_project.results:
            QMessageBox.information(self, "No Results", "There are no analysis results to export.")
            return
        
        # Get export location
        export_file, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", 
            "PDF Reports (*.pdf);;HTML Reports (*.html);;CSV Data (*.csv);;JSON Data (*.json);;All Files (*)"
        )
        
        if export_file:
            try:
                # Export based on file type
                if export_file.endswith('.pdf'):
                    self.project_manager.export_pdf_report(export_file)
                elif export_file.endswith('.html'):
                    self.project_manager.export_html_report(export_file)
                elif export_file.endswith('.csv'):
                    self.project_manager.export_csv_data(export_file)
                elif export_file.endswith('.json'):
                    self.project_manager.export_json_data(export_file)
                
                self.statusBar().showMessage(f"Results exported to {export_file}")
            except Exception as e:
                QMessageBox.warning(self, "Export Error", f"Failed to export results: {str(e)}")
    
    def _on_run_analysis(self):
        """Handle running crowd analysis"""
        if not self.project_manager.current_project or not self.project_manager.current_dataset:
            QMessageBox.information(self, "No Data", "Please select a dataset for analysis.")
            return
        
        # Get analysis parameters from panel
        params = self.analysis_panel.get_parameters()
        
        try:
            # Run the analysis
            self.statusBar().showMessage("Running analysis...")
            
            # This would typically be run in a separate thread
            # For now, we'll simulate it with a direct call
            results = self.project_manager.run_analysis(params)
            
            # Update UI with results
            self.results_panel.display_results(results)
            
            # Update visualization
            if 'density_map' in results:
                self.visualization_widget.display_density_map(results['density_map'])
            
            self.statusBar().showMessage("Analysis complete")
        except Exception as e:
            QMessageBox.warning(self, "Analysis Error", f"Failed to run analysis: {str(e)}")
    
    def _on_generate_report(self):
        """Handle generating a comprehensive report"""
        if not self.project_manager.current_project or not hasattr(self.project_manager.current_project, 'results') or not self.project_manager.current_project.results:
            QMessageBox.information(self, "No Results", "Run an analysis first to generate a report.")
            return
        
        # Get report location
        report_file, _ = QFileDialog.getSaveFileName(
            self, "Generate Report", "", "PDF Reports (*.pdf);;HTML Reports (*.html);;All Files (*)"
        )
        
        if report_file:
            try:
                # Generate report based on file type
                if report_file.endswith('.pdf'):
                    self.project_manager.generate_pdf_report(report_file)
                elif report_file.endswith('.html'):
                    self.project_manager.generate_html_report(report_file)
                
                self.statusBar().showMessage(f"Report generated at {report_file}")
            except Exception as e:
                QMessageBox.warning(self, "Report Error", f"Failed to generate report: {str(e)}")
    
    def _on_project_selected(self, project_id):
        """Handle project selection in project explorer"""
        project = self.project_manager.get_project(project_id)
        if project:
            self.project_manager.current_project = project
            self.status_project.setText(f"Project: {project.name}")
    
    def _on_dataset_selected(self, dataset_id):
        """Handle dataset selection in project explorer"""
        dataset = self.project_manager.get_dataset(dataset_id)
        if dataset:
            self.project_manager.current_dataset = dataset
            self.visualization_widget.display_point_cloud(dataset.points)
            self.status_points.setText(f"Points: {len(dataset.points):,}")
    
    def _on_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, 
            "About LiDAR Crowd Analytics",
            f"<h3>LiDAR Crowd Analytics</h3>"
            f"<p>Version 1.0.0</p>"
            f"<p>A professional tool for analyzing crowd density and flow patterns "
            f"using LiDAR point cloud data.</p>"
            f"<p>Copyright © {datetime.now().year}</p>"
        )
    
    def closeEvent(self, event):
        """Handle application close event"""
        if self.project_manager.current_project and self.project_manager.current_project.modified:
            reply = QMessageBox.question(
                self, 
                "Unsaved Changes", 
                "The current project has unsaved changes. Save before closing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Save:
                # Save project
                saved = self._on_save_project()
                if not saved:
                    event.ignore()
                    return
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
        
        # Close database connection
        self.db_manager.close()
        
        # Accept the close event
        event.accept()


def main():
    """Application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("LiDAR Crowd Analytics")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Your Organization")
    app.setOrganizationDomain("yourorganization.com")
    
    # Create and show the main window
    window = MainWindow()
    window.show()
    
    # Start the event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()