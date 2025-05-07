"""
LiDAR-based Crowd Management System
"""

__version__ = '1.0.0'

from .detector import CrowdDetector
from .analyzer import CrowdAnalyzer
from .visualizer import CrowdVisualizer
from .strategies import CrowdManager
from .config import DEFAULT_CONFIG

__all__ = ['CrowdDetector', 'CrowdAnalyzer', 'CrowdVisualizer', 'CrowdManager', 'DEFAULT_CONFIG'] 