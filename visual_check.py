import laspy
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
from laspy import LazBackend

def read_laz_file(laz_path):
    """Read LAZ file with fallback options"""
    try:
        # Try laszip backend first
        las = laspy.read(laz_path, laz_backend=LazBackend.Laszip)
        print("Using Laszip backend")
        return las
    except Exception as e1:
        print(f"Laszip backend failed: {e1}")
        try:
            # Try single-threaded lazrs
            las = laspy.read(laz_path, laz_backend=LazBackend.Lazrs)
            print("Using Lazrs backend (single-threaded)")
            return las
        except Exception as e2:
            print(f"Lazrs backend failed: {e2}")
            try:
                # Last resort: try with automatic backend selection
                las = laspy.read(laz_path)
                print("Using automatic backend selection")
                return las
            except Exception as e3:
                raise Exception(f"Failed to read LAZ file with any backend: {e3}")

def visualize_forest_data(laz_path, gpkg_path):
    """Create comprehensive visualizations of the forest LiDAR data"""

    # Read LAZ file with robust error handling
    las = read_laz_file(laz_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()

    # Read boundary
    boundary = gpd.read_file(gpkg_path)

    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))

    # 1. Height distribution
    ax1 = fig.add_subplot(221)
    sns.histplot(las.z, bins=50, ax=ax1)
    ax1.set_title('Height Distribution')
    ax1.set_xlabel('Height (m)')
    ax1.set_ylabel('Count')

    # 2. 2D point density
    ax2 = fig.add_subplot(222)
    h = ax2.hist2d(las.x, las.y, bins=100, cmap='viridis')
    plt.colorbar(h[3], ax=ax2)
    ax2.set_title('Point Density (2D View)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')

    # 3. 3D scatter plot (sample)
    ax3 = fig.add_subplot(223, projection='3d')
    # Sample points for visualization (e.g., 1% of points)
    sample_size = len(las.x) // 100
    idx = np.random.choice(len(las.x), sample_size, replace=False)
    scatter = ax3.scatter(las.x[idx], las.y[idx], las.z[idx],
                         c=las.z[idx], cmap='viridis', s=1)
    plt.colorbar(scatter, ax=ax3)
    ax3.set_title('3D Point Cloud (1% sample)')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_zlabel('Height (m)')

    # 4. Height profile
    ax4 = fig.add_subplot(224)
    x_center = (las.x.min() + las.x.max()) / 2
    mask = (las.x > x_center - 1) & (las.x < x_center + 1)
    ax4.scatter(las.y[mask], las.z[mask], s=1, alpha=0.1)
    ax4.set_title('Forest Profile (1m slice)')
    ax4.set_xlabel('Y (m)')
    ax4.set_ylabel('Height (m)')

    plt.tight_layout()
    plt.savefig('forest_visualization.png', dpi=300, bbox_inches='tight')

    # Print summary statistics
    print("\n=== Forest Stand Statistics ===")
    print(f"Average height: {las.z.mean():.2f}m")
    print(f"Maximum height: {las.z.max():.2f}m")
    print(f"Point density: {len(las.x) / boundary.area.iloc[0]:.2f} points/m²")
    print(f"Stand area: {boundary.area.iloc[0]:.2f}m²")

    # Create height percentile table
    percentiles = np.percentile(las.z, [0, 25, 50, 75, 100])
    print("\n=== Height Percentiles ===")
    print(f"0th (min): {percentiles[0]:.2f}m")
    print(f"25th: {percentiles[1]:.2f}m")
    print(f"50th (median): {percentiles[2]:.2f}m")
    print(f"75th: {percentiles[3]:.2f}m")
    print(f"100th (max): {percentiles[4]:.2f}m")

if __name__ == "__main__":
    laz_path = "data/als_data.laz"
    gpkg_path = "data/stand_boundary.gpkg"
    visualize_forest_data(laz_path, gpkg_path)
