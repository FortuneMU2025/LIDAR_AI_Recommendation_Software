import os
import numpy as np
import pandas as pd

def compute_box_volume(min_pt, max_pt):
    return np.prod(np.abs(max_pt[:3] - min_pt[:3]))

def analyze_processed_folder(processed_dir):
    summary = []
    for fname in sorted(os.listdir(processed_dir)):
        if fname.endswith('_clusters.npy'):
            base = fname.replace('_clusters.npy', '')
            clusters = np.load(os.path.join(processed_dir, fname))
            points = np.load(os.path.join(processed_dir, f"{base}_processed.npy"))
            n_clusters = len(np.unique(clusters[clusters != -1]))
            # Compute bounding box volumes
            volumes = []
            for label in np.unique(clusters):
                if label == -1:
                    continue
                cluster_points = points[clusters == label]
                min_pt = cluster_points.min(axis=0)
                max_pt = cluster_points.max(axis=0)
                volumes.append(compute_box_volume(min_pt, max_pt))
            if volumes:
                avg_vol = np.mean(volumes)
                min_vol = np.min(volumes)
                max_vol = np.max(volumes)
            else:
                avg_vol = min_vol = max_vol = 0
            summary.append({
                'frame': base,
                'n_clusters': n_clusters,
                'avg_box_vol': avg_vol,
                'min_box_vol': min_vol,
                'max_box_vol': max_vol
            })
    df = pd.DataFrame(summary)
    print(df)
    # Save to CSV for further inspection
    df.to_csv(os.path.join(processed_dir, 'bounding_box_stats.csv'), index=False)
    print(f"\nSummary saved to {os.path.join(processed_dir, 'bounding_box_stats.csv')}")
    # Print some global stats
    print("\nGlobal stats:")
    print(f"Frames analyzed: {len(df)}")
    print(f"Average clusters per frame: {df['n_clusters'].mean():.2f}")
    print(f"Average bounding box volume: {df['avg_box_vol'].mean():.2f}")
    print(f"Min bounding box volume: {df['min_box_vol'].min():.2f}")
    print(f"Max bounding box volume: {df['max_box_vol'].max():.2f}")

if __name__ == "__main__":
    processed_dir = "kitti_dataset/processed"  # Change if needed
    analyze_processed_folder(processed_dir) 