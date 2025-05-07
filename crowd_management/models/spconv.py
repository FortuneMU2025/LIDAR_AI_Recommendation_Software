"""
Sparse convolution network implementation for crowd detection.
"""

import torch
import torch.nn as nn
import spconv.pytorch as spconv
from typing import Dict, Tuple

class SparseConvDetector(nn.Module):
    """Sparse convolution network for crowd detection."""
    
    def __init__(self):
        super().__init__()
        
        # Voxel feature extraction
        self.vfe = VoxelFeatureExtractor(4, 16)
        
        # Sparse convolution backbone
        self.backbone = nn.ModuleDict({
            'block1': self._make_sparse_block(16, 32, 2),
            'block2': self._make_sparse_block(32, 64, 2),
            'block3': self._make_sparse_block(64, 128, 2),
            'block4': self._make_sparse_block(128, 256, 2)
        })
        
        # Detection heads
        self.cls_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        
        self.box_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 7)  # (x, y, z, dx, dy, dz, heading)
        )
        
    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            points: (B, N, 4) tensor of point coordinates and features
            
        Returns:
            Dictionary containing detection results
        """
        # Convert points to sparse tensor
        voxels, indices = self._voxelize(points)
        batch_size = points.shape[0]
        
        # Extract voxel features
        voxel_features = self.vfe(voxels)
        
        # Create sparse tensor
        spatial_shape = self._compute_spatial_shape(indices)
        x = spconv.SparseConvTensor(
            features=voxel_features,
            indices=indices,
            spatial_shape=spatial_shape,
            batch_size=batch_size
        )
        
        # Backbone forward
        for block in self.backbone.values():
            x = block(x)
            
        # Get dense features
        features = x.dense()
        features = features.mean(dim=(2, 3, 4))  # Global average pooling
        
        # Detection heads
        cls_scores = self.cls_head(features).squeeze(-1)
        box_preds = self.box_head(features)
        
        # Post-processing
        cls_probs = torch.sigmoid(cls_scores)
        boxes = self._decode_boxes(box_preds)
        
        return {
            'scores': cls_probs,
            'boxes': boxes
        }
        
    def _make_sparse_block(self, in_channels: int, out_channels: int,
                           stride: int = 1) -> nn.Module:
        """Create a sparse convolution block."""
        return spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            spconv.SparseConv3d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
    def _voxelize(self, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert points to voxels."""
        # Voxel size: 0.1m
        voxel_size = torch.tensor([0.1, 0.1, 0.1], device=points.device)
        
        # Compute voxel indices
        voxel_coords = torch.floor(points[..., :3] / voxel_size).int()
        
        # Get unique voxels
        voxels = []
        indices = []
        batch_indices = []
        
        for b in range(points.shape[0]):
            batch_mask = torch.ones(points.shape[1], dtype=torch.bool, device=points.device)
            batch_points = points[b]
            batch_coords = voxel_coords[b]
            
            # Get unique voxels
            unique_coords, inverse_indices = torch.unique(batch_coords, dim=0, return_inverse=True)
            
            # Aggregate points in each voxel
            for i in range(len(unique_coords)):
                voxel_mask = inverse_indices == i
                voxel_points = batch_points[voxel_mask]
                
                if len(voxel_points) > 0:
                    voxels.append(voxel_points.mean(dim=0))
                    indices.append(unique_coords[i])
                    batch_indices.append(b)
                    
        voxels = torch.stack(voxels)
        indices = torch.stack(indices)
        batch_indices = torch.tensor(batch_indices, device=points.device)
        
        # Add batch dimension to indices
        indices = torch.cat([batch_indices.unsqueeze(-1), indices], dim=-1)
        
        return voxels, indices
        
    def _compute_spatial_shape(self, indices: torch.Tensor) -> Tuple[int, int, int]:
        """Compute spatial shape from indices."""
        max_indices = indices[:, 1:].max(dim=0)[0]
        return tuple((max_indices + 1).cpu().numpy().tolist())
        
    def _decode_boxes(self, box_preds: torch.Tensor) -> torch.Tensor:
        """Decode box parameters into corner coordinates."""
        # Extract parameters
        xyz = box_preds[..., :3]
        dims = torch.exp(box_preds[..., 3:6])  # positive dimensions
        heading = box_preds[..., 6]
        
        # Create corner coordinates
        corners = torch.zeros_like(xyz).repeat(1, 8, 1)
        
        # Unit cube corners
        unit_corners = torch.tensor([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
        ], device=xyz.device) * 0.5
        
        # Scale and rotate corners
        for i in range(xyz.shape[0]):
            R = self._rotation_matrix(heading[i])
            corners[i] = torch.matmul(unit_corners * dims[i], R.T) + xyz[i]
            
        return corners
        
    @staticmethod
    def _rotation_matrix(heading: torch.Tensor) -> torch.Tensor:
        """Create 3D rotation matrix from heading angle."""
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        
        return torch.tensor([
            [cos_h, -sin_h, 0],
            [sin_h, cos_h, 0],
            [0, 0, 1]
        ], device=heading.device)

class VoxelFeatureExtractor(nn.Module):
    """Extract features from voxelized points."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(voxels) 