# src/visualization.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import geopandas as gpd

class ForestVisualizer:
    def __init__(self):
        # Set up plotting style
        plt.style.use('default')  # Use default style instead of seaborn
        plt.rcParams.update({
            'figure.figsize': (10, 8),
            'figure.dpi': 300,
            'font.family': 'sans-serif',
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.axisbelow': True,
            'axes.labelpad': 10,
            'axes.titlepad': 20
        })

    def plot_boundary_with_points(self, x, y, z, boundary_gdf, title="Point Cloud with Forest Boundary"):
        """Plot points with forest boundary overlay"""
        try:
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 10))

            # Plot points
            scatter = ax.scatter(x, y, c=z,
                               cmap='viridis',
                               s=1, alpha=0.5,
                               label='LiDAR Points')

            # Plot boundary
            if boundary_gdf is not None:
                boundary_gdf.boundary.plot(ax=ax,
                                         color='red',
                                         linewidth=2,
                                         label='Forest Boundary')

            # Formatting
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Height (m)', rotation=90, labelpad=15)
            ax.set_title(title, pad=20, weight='bold')
            ax.set_xlabel('X coordinate (m)', labelpad=10)
            ax.set_ylabel('Y coordinate (m)', labelpad=10)
            ax.legend(loc='upper right')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

            return fig, ax

        except Exception as e:
            print(f"Error in boundary plot creation: {str(e)}")
            return None, None

    def plot_height_distribution(self, heights, ground_level=None, title="Height Distribution"):
        """Plot height distribution with statistics"""
        try:
            # Create histogram
            plt.hist(heights, bins=50,
                    color='forestgreen',
                    alpha=0.7,
                    edgecolor='black',
                    linewidth=0.5,
                    density=False)  # Set density=False to show actual frequencies

            # Calculate statistics
            mean_height = np.mean(heights)
            std_height = np.std(heights)
            median_height = np.median(heights)

            # Add reference lines
            plt.axvline(mean_height, color='red', linestyle='--',
                       label=f'Mean: {mean_height:.1f} m')
            plt.axvline(median_height, color='blue', linestyle=':',
                       label=f'Median: {median_height:.1f} m')
            plt.axvline(mean_height + std_height, color='gray', linestyle=':',
                       label=f'±1 SD: {std_height:.1f} m')
            plt.axvline(mean_height - std_height, color='gray', linestyle=':')

            if ground_level is not None:
                plt.axvline(ground_level, color='green', linestyle='-',
                           label=f'Ground Level: {ground_level:.1f} m')

            # Set labels and title
            plt.title(title, pad=20, weight='bold')
            plt.xlabel('Height (m)', labelpad=10)
            plt.ylabel('Frequency', labelpad=10)
            plt.legend(frameon=True, facecolor='white', framealpha=1)
            plt.grid(True, alpha=0.3)

            # Adjust y-axis to show data clearly
            counts, bins = np.histogram(heights, bins=50)
            plt.ylim(0, max(counts) * 1.1)  # Add 10% margin at the top

        except Exception as e:
            print(f"Error in height distribution plot: {str(e)}")

    def plot_point_cloud_2d(self, x, y, z, title="2D Point Cloud View", sample_rate=10):
        """Create 2D top-down view of point cloud"""
        try:
            # Sample points for better visualization
            scatter = plt.scatter(x[::sample_rate],
                                y[::sample_rate],
                                c=z[::sample_rate],
                                cmap='viridis',
                                s=1,
                                alpha=0.6)

            plt.colorbar(scatter, label='Height (m)')
            plt.title(title, pad=20, weight='bold')
            plt.xlabel('X coordinate (m)', labelpad=10)
            plt.ylabel('Y coordinate (m)', labelpad=10)
            plt.grid(True, alpha=0.3)

        except Exception as e:
            print(f"Error in 2D plot creation: {str(e)}")

    def plot_3d_sample(self, x, y, z, sample_size=10000):
        """Create 3D visualization of point cloud sample"""
        try:
            # Sample points if necessary
            if len(x) > sample_size:
                idx = np.random.choice(len(x), sample_size, replace=False)
            else:
                idx = slice(None)

            # Create 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            scatter = ax.scatter(x[idx], y[idx], z[idx],
                               c=z[idx],
                               cmap='viridis',
                               s=1)

            # Formatting
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Height (m)', rotation=90, labelpad=15)

            ax.set_title('3D Point Cloud Sample', pad=20, weight='bold')
            ax.set_xlabel('X (m)', labelpad=10)
            ax.set_ylabel('Y (m)', labelpad=10)
            ax.set_zlabel('Height (m)', labelpad=10)
            ax.view_init(elev=30, azim=45)
            ax.grid(True, alpha=0.3)

            return fig, ax

        except Exception as e:
            print(f"Error in 3D plot creation: {str(e)}")
            return None, None

def create_analysis_figure(processor, detector, visualizer):
    """Create comprehensive analysis figures"""
    try:
        # Figure 1: Point Cloud Analysis
        fig1 = plt.figure(figsize=(15, 5))
        plt.suptitle('Point Cloud Analysis', fontsize=16, y=0.95)

        # Plot 1: Points with boundary
        ax1 = plt.subplot(131)
        scatter1 = ax1.scatter(processor.x[::10], processor.y[::10],
                             c=processor.z[::10],
                             cmap='viridis',
                             s=1, alpha=0.6)
        if processor.boundary is not None:
            processor.boundary.boundary.plot(ax=ax1,
                                          color='red',
                                          linewidth=2,
                                          label='Forest Boundary')
        plt.colorbar(scatter1, ax=ax1, label='Height (m)')
        ax1.set_title('Raw Point Cloud')
        ax1.set_xlabel('X coordinate (m)')
        ax1.set_ylabel('Y coordinate (m)')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Normalized Point Cloud
        ax2 = plt.subplot(132)
        scatter2 = ax2.scatter(processor.x[::10], processor.y[::10],
                             c=processor.z_normalized[::10],
                             cmap='viridis',
                             s=1, alpha=0.6)
        plt.colorbar(scatter2, ax=ax2, label='Normalized Height (m)')
        ax2.set_title('Normalized Point Cloud')
        ax2.set_xlabel('X coordinate (m)')
        ax2.set_ylabel('Y coordinate (m)')
        ax2.grid(True, alpha=0.3)

        # Plot 3: 3D Sample
        ax3 = plt.subplot(133, projection='3d')
        sample_size = min(10000, len(processor.x))
        idx = np.random.choice(len(processor.x), sample_size, replace=False)
        scatter3 = ax3.scatter(processor.x[idx],
                             processor.y[idx],
                             processor.z_normalized[idx],
                             c=processor.z_normalized[idx],
                             cmap='viridis',
                             s=1, alpha=0.6)
        plt.colorbar(scatter3, ax=ax3, label='Height (m)')
        ax3.set_title('3D Point Cloud Sample')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_zlabel('Height (m)')
        ax3.view_init(elev=30, azim=45)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig1.savefig('point_cloud_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig1)

        # Figure 2: Tree Analysis
        fig2 = plt.figure(figsize=(15, 5))
        plt.suptitle('Tree Analysis Results', fontsize=16, y=0.95)

        # Plot 1: Detected Trees
        ax4 = plt.subplot(131)
        if detector.trees_detected:
            scatter4 = ax4.scatter(detector.tree_positions[:, 0],
                                 detector.tree_positions[:, 1],
                                 c=detector.tree_heights,
                                 cmap='viridis',
                                 s=50, alpha=0.8)
            plt.colorbar(scatter4, ax=ax4, label='Tree Height (m)')
            if processor.boundary is not None:
                processor.boundary.boundary.plot(ax=ax4,
                                              color='red',
                                              linewidth=2,
                                              label='Forest Boundary')
        ax4.set_title('Detected Trees')
        ax4.set_xlabel('X coordinate (m)')
        ax4.set_ylabel('Y coordinate (m)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()

        # Plot 2: Height Distribution
        ax5 = plt.subplot(132)
        ax5.hist(processor.z_normalized,
                bins=50,
                color='forestgreen',
                alpha=0.7,
                edgecolor='black')
        ax5.axvline(np.mean(processor.z_normalized),
                    color='red',
                    linestyle='--',
                    label=f'Mean Height')
        ax5.set_title('Height Distribution')
        ax5.set_xlabel('Height (m)')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Plot 3: Diameter Distribution
        ax6 = plt.subplot(133)
        if hasattr(detector, 'tree_diameters'):
            diameters_cm = detector.tree_diameters * 100
            ax6.hist(diameters_cm,
                    bins=30,
                    color='forestgreen',
                    alpha=0.7,
                    edgecolor='black')
            ax6.axvline(np.mean(diameters_cm),
                       color='red',
                       linestyle='--',
                       label=f'Mean DBH')
            ax6.set_title('Tree Diameter Distribution')
            ax6.set_xlabel('DBH (cm)')
            ax6.set_ylabel('Frequency')
            ax6.legend()
            ax6.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig2.savefig('tree_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)

        return True

    except Exception as e:
        print(f"Error creating analysis figures: {str(e)}")
        return False