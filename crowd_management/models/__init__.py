"""
Neural network models for LiDAR-based crowd detection.
"""
from .pointnet2 import PointNet2Detector
from .spconv import SparseConvDetector
from .cylinder3d import Cylinder3DDetector
def get_model(model_type: str):
    """
    Get the neural network model based on type.
    
    Args:
        model_type: Type of model to use ('pointnet2', 'spconv', or 'cylinder3d')
        
    Returns:
        Neural network model instance
    """
    models = {
        'pointnet2': PointNet2Detector,
        'spconv': SparseConvDetector,
        'cylinder3d': Cylinder3DDetector
    }
    
    assert model_type in models, f"Model type {model_type} not supported. Choose from {list(models.keys())}"
    return models[model_type]() 