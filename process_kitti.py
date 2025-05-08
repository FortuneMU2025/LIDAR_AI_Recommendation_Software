import os
import numpy as np
from crowd_management.pipeline import process_point_cloud
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_kitti_lidar(file_path):
    """Read KITTI LiDAR data from binary file."""
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points

def get_bounding_boxes(points, labels, min_points=10, min_vol=0.01, max_vol=500):
    """Compute axis-aligned 3D bounding boxes for each cluster, with filtering."""
    boxes = []
    for label in np.unique(labels):
        if label == -1:
            continue  # Skip noise
        cluster_points = points[labels == label]
        if len(cluster_points) < min_points:
            continue  # Skip small clusters
        min_pt = cluster_points.min(axis=0)
        max_pt = cluster_points.max(axis=0)
        vol = np.prod(np.abs(max_pt[:3] - min_pt[:3]))
        if vol < min_vol or vol > max_vol:
            continue  # Skip tiny or huge boxes
        boxes.append((min_pt, max_pt))
    return boxes

def draw_3d_box(ax, min_pt, max_pt, color='cyan'):
    """Draw a 3D bounding box on the given axes."""
    # 8 corners of the box
    corners = np.array([
        [min_pt[0], min_pt[1], min_pt[2]],
        [min_pt[0], min_pt[1], max_pt[2]],
        [min_pt[0], max_pt[1], min_pt[2]],
        [min_pt[0], max_pt[1], max_pt[2]],
        [max_pt[0], min_pt[1], min_pt[2]],
        [max_pt[0], min_pt[1], max_pt[2]],
        [max_pt[0], max_pt[1], min_pt[2]],
        [max_pt[0], max_pt[1], max_pt[2]],
    ])
    # List of box edges
    edges = [
        [0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],
        [4,5],[4,6],[5,7],[6,7]
    ]
    for edge in edges:
        ax.plot3D(*zip(corners[edge[0]], corners[edge[1]]), color=color, linewidth=1)

def visualize_point_cloud(points, clusters=None, show_boxes=True, save_path=None):
    """Visualize point cloud with optional cluster coloring and bounding boxes."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if clusters is not None:
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                           c=clusters, cmap='viridis', s=1)
        plt.colorbar(scatter, label='Cluster ID')
        if show_boxes:
            boxes = get_bounding_boxes(points, clusters)
            for min_pt, max_pt in boxes:
                draw_3d_box(ax, min_pt, max_pt, color='cyan')
    else:
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                           c=points[:, 3], cmap='viridis', s=1)
        plt.colorbar(scatter, label='Intensity')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.title('KITTI Point Cloud Visualization')
    max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                         points[:, 1].max()-points[:, 1].min(),
                         points[:, 2].max()-points[:, 2].min()]).max() / 2.0
    mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def process_kitti_file(file_path, config_path=None, save_dir=None):
    """Process a single KITTI LiDAR file."""
    points = read_kitti_lidar(file_path)
    print(f"Read {len(points)} points from file {os.path.basename(file_path)}")
    processed_points, clusters = process_point_cloud(points, config_path)
    print(f"Processed {len(processed_points)} points into {len(np.unique(clusters))} clusters")
    if save_dir:
        base = os.path.splitext(os.path.basename(file_path))[0]
        np.save(os.path.join(save_dir, f"{base}_processed.npy"), processed_points)
        np.save(os.path.join(save_dir, f"{base}_clusters.npy"), clusters)
        # Save visualization image
        img_path = os.path.join(save_dir, f"{base}_viz.png")
        visualize_point_cloud(processed_points, clusters, show_boxes=True, save_path=img_path)
    else:
        visualize_point_cloud(processed_points, clusters, show_boxes=True)
    return processed_points, clusters

def batch_process_kitti_folder(folder_path, config_path=None, save_dir=None, visualize_first=True):
    files = [f for f in os.listdir(folder_path) if f.endswith('.bin')]
    files.sort()
    print(f"Found {len(files)} .bin files in {folder_path}")
    summary = []
    for idx, file in enumerate(files):
        file_path = os.path.join(folder_path, file)
        processed_points, clusters = process_kitti_file(file_path, config_path, save_dir)
        summary.append((file, len(processed_points), len(np.unique(clusters))))
        if visualize_first and idx == 0 and not save_dir:
            visualize_point_cloud(processed_points, clusters, show_boxes=True)
    print("\nBatch processing summary:")
    for file, n_points, n_clusters in summary:
        print(f"{file}: {n_points} points, {n_clusters} clusters")
    print("Done.")

def main():
    data_dir = "kitti_dataset/training/velodyne"
    save_dir = "kitti_dataset/processed"  # Change or set to None if you don't want to save
    os.makedirs(save_dir, exist_ok=True)
    batch_process_kitti_folder(data_dir, save_dir=save_dir, visualize_first=True)

if __name__ == "__main__":
    main() 