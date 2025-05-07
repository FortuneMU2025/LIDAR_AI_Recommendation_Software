"""
PointNet++ implementation for crowd detection from LiDAR point clouds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class SAModule(nn.Module):
    """Set abstraction module for PointNet++."""
    
    def __init__(self, ratio: float, r: float, nn: List[Tuple[int, int]]):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = nn.ModuleList()
        
        for i, (in_channel, out_channel) in enumerate(nn):
            self.conv.append(nn.Conv1d(in_channel, out_channel, 1))
            if i != len(nn) - 1:
                self.conv.append(nn.ReLU())
                
    def forward(self, xyz: torch.Tensor, points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Set abstraction forward pass.
        
        Args:
            xyz: (B, N, 3) tensor of point coordinates
            points: (B, C, N) tensor of point features
            
        Returns:
            new_xyz: (B, N', 3) tensor of subsampled point coordinates
            new_points: (B, C', N') tensor of updated point features
        """
        # FPS sampling
        fps_idx = farthest_point_sample(xyz, int(xyz.shape[1] * self.ratio))
        new_xyz = index_points(xyz, fps_idx)
        
        # Ball query grouping
        idx = query_ball_point(self.r, xyz.shape[1], new_xyz, xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_xyz -= new_xyz.unsqueeze(2)
        
        if points is not None:
            grouped_points = index_points(points.transpose(1, 2), idx)
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz
            
        # Point feature embedding
        grouped_points = grouped_points.permute(0, 3, 1, 2)
        for conv in self.conv:
            grouped_points = conv(grouped_points)
        new_points = torch.max(grouped_points, -1)[0]
        
        return new_xyz, new_points

class PointNet2Detector(nn.Module):
    """PointNet++ for crowd detection."""
    
    def __init__(self):
        super().__init__()
        
        # Set abstraction layers
        self.sa1 = SAModule(0.5, 0.1, [(3, 64), (64, 64), (64, 128)])
        self.sa2 = SAModule(0.25, 0.2, [(128, 128), (128, 128), (128, 256)])
        self.sa3 = SAModule(0.25, 0.4, [(256, 256), (256, 256), (256, 512)])
        
        # Feature propagation layers
        self.fp3 = FeaturePropagation(768, [256, 256])
        self.fp2 = FeaturePropagation(384, [256, 128])
        self.fp1 = FeaturePropagation(128, [128, 128, 128])
        
        # Detection heads
        self.cls_head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, 1, 1)  # Binary classification
        )
        
        self.box_head = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(128, 7, 1)  # (x, y, z, dx, dy, dz, heading)
        )
        
    def forward(self, points: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            points: (B, N, 4) tensor of point coordinates and features
            
        Returns:
            Dictionary containing detection results
        """
        xyz = points[..., :3]
        features = points[..., 3:].transpose(1, 2)
        
        # Set abstraction
        xyz1, features1 = self.sa1(xyz, features)
        xyz2, features2 = self.sa2(xyz1, features1)
        xyz3, features3 = self.sa3(xyz2, features2)
        
        # Feature propagation
        features2 = self.fp3(xyz2, xyz3, features2, features3)
        features1 = self.fp2(xyz1, xyz2, features1, features2)
        features = self.fp1(xyz, xyz1, None, features1)
        
        # Detection heads
        cls_scores = self.cls_head(features).squeeze(1)
        box_preds = self.box_head(features).transpose(1, 2)
        
        # Post-processing
        cls_probs = torch.sigmoid(cls_scores)
        boxes = self._decode_boxes(box_preds)
        
        return {
            'scores': cls_probs,
            'boxes': boxes
        }
        
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

class FeaturePropagation(nn.Module):
    """Feature propagation layer."""
    
    def __init__(self, in_channel: int, mlp: List[int]):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
            
    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor,
                points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
        """
        Propagate features from xyz2 to xyz1.
        
        Args:
            xyz1: (B, N1, 3) coordinates of the first points
            xyz2: (B, N2, 3) coordinates of the second points
            points1: (B, C1, N1) features of the first points
            points2: (B, C2, N2) features of the second points
            
        Returns:
            new_points: (B, mlp[-1], N1) propagated features
        """
        if xyz2 is None:
            interpolated_points = points2
        else:
            dists, idx = three_nn(xyz1, xyz2)
            dists = torch.clamp(dists, min=1e-10)
            weights = 1.0 / dists
            weights = weights / torch.sum(weights, dim=-1, keepdim=True)
            interpolated_points = three_interpolate(points2, idx, weights)
            
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)
        else:
            new_points = interpolated_points
            
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
            
        return new_points

# Utility functions for point cloud operations
def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest point sampling.
    
    Args:
        xyz: (B, N, 3) input points
        npoint: Number of points to sample
        
    Returns:
        centroids: (B, npoint) indices of sampled points
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
        
    return centroids

def query_ball_point(radius: float, nsample: int, xyz: torch.Tensor,
                     new_xyz: torch.Tensor) -> torch.Tensor:
    """
    Find points within radius.
    
    Args:
        radius: Ball query radius
        nsample: Maximum number of points to sample
        xyz: (B, N, 3) all points
        new_xyz: (B, S, 3) query points
        
    Returns:
        group_idx: (B, S, nsample) grouped points indices
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Calculate squared distance between each two points.
    
    Args:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Index points by given indices.
    
    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] or [B, S, nsample]
        
    Returns:
        new_points: indexed points data, [B, S, C] or [B, S, nsample, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def three_nn(unknown: torch.Tensor, known: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the three nearest neighbors.
    
    Args:
        unknown: (B, N, 3)
        known: (B, M, 3)
        
    Returns:
        dist: (B, N, 3) squared distances
        idx: (B, N, 3) indices of nearest neighbors
    """
    dists = square_distance(unknown, known)
    dists, idx = torch.sort(dists, dim=-1)
    return dists[:, :, :3], idx[:, :, :3]

def three_interpolate(points: torch.Tensor, idx: torch.Tensor,
                      weight: torch.Tensor) -> torch.Tensor:
    """
    Interpolate point features from known points to unknown points.
    
    Args:
        points: (B, C, M) known features
        idx: (B, N, 3) indices of nearest neighbors
        weight: (B, N, 3) weights
        
    Returns:
        new_points: (B, C, N) interpolated features
    """
    B, C, M = points.shape
    _, N, _ = idx.shape
    
    points_view = points.unsqueeze(2).expand(B, C, N, 3)
    batch_indices = torch.arange(B).view(B, 1, 1).expand(B, N, 3).to(points.device)
    idx_view = idx.unsqueeze(1).expand(B, C, N, 3)
    
    weight_view = weight.unsqueeze(1).expand(B, C, N, 3)
    interpolated_points = torch.sum(points_view * weight_view, dim=-1)
    
    return interpolated_points 