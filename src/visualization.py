# src/visualization.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

class ForestVisualizer:
    def __init__(self):
        # Change this line to use a valid style
        plt.style.use('seaborn-v0_8')  # or use 'seaborn-darkgrid' depending on your seaborn version
        sns.set_theme()  # This will set the seaborn defaults

    def plot_point_cloud_2d(self, x, y, z, title="2D Point Cloud View", sample_rate=100):
        """
        Create a 2D top-down view of the point cloud
        sample_rate: plot every nth point to manage memory
        """
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(x[::sample_rate], y[::sample_rate],
                            c=z[::sample_rate],
                            cmap='viridis',
                            s=1, alpha=0.5)
        plt.colorbar(scatter, label='Height (m)')
        plt.title(title)
        plt.xlabel('X coordinate (m)')
        plt.ylabel('Y coordinate (m)')
        plt.show()

    def plot_height_distribution(self, z):
        """
        Plot height distribution of points
        """
        plt.figure(figsize=(10, 6))
        plt.hist(z, bins=50, color='forestgreen', alpha=0.7)
        plt.title('Height Distribution of Points')
        plt.xlabel('Height (m)')
        plt.ylabel('Frequency')
        plt.show()

    def plot_3d_sample(self, x, y, z, sample_size=10000):
        """
        Create 3D visualization of point cloud sample
        """
        idx = np.random.randint(0, len(x), sample_size)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(x[idx], y[idx], z[idx],
                           c=z[idx],
                           cmap='viridis',
                           s=1)
        plt.colorbar(scatter, label='Height (m)')
        ax.set_title('3D Point Cloud Sample')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Height (m)')
        plt.show()
