# src/tree_detection.py

import numpy as np
from scipy.ndimage import maximum_filter, gaussian_filter
import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas as gpd
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from scipy.stats import norm
import pandas as pd
class TreeDetector:
    def __init__(self, window_size=5, min_height=2):
        
        self.window_size = window_size
        self.min_height = min_height
        # Tree metrics
        self.tree_positions = None
        self.tree_heights = None
        self.tree_diameters = None
        self.basal_area = None
        # Processing flags
        self.trees_detected = False
        self.dbh_calculated = False
        self.basal_area_calculated = False
       
    def detect_trees(self, x, y, z):
        """
        Detect trees using local maxima detection

        Parameters:
        -----------
        x, y : array-like
            Coordinates of points
        z : array-like
            Normalized heights of points

        Returns:
        --------
        tree_positions : ndarray
            Array of tree positions (x, y)
        tree_heights : ndarray
            Array of tree heights
        """
        try:
            print("Starting tree detection...")

            if len(x) == 0 or len(y) == 0 or len(z) == 0:
                raise ValueError("Empty input arrays")

            # Grid parameters
            x_res = 1.0  # 1m resolution
            y_res = 1.0

            # Create grid
            x_grid = np.arange(min(x), max(x) + x_res, x_res)
            y_grid = np.arange(min(y), max(y) + y_res, y_res)
            grid_height = np.full((len(y_grid), len(x_grid)), np.nan)

            # Fill grid with maximum heights
            x_idx = np.clip(((x - min(x)) / x_res).astype(int), 0, len(x_grid)-1)
            y_idx = np.clip(((y - min(y)) / y_res).astype(int), 0, len(y_grid)-1)

            for i in range(len(x)):
                current_height = grid_height[y_idx[i], x_idx[i]]
                if np.isnan(current_height) or z[i] > current_height:
                    grid_height[y_idx[i], x_idx[i]] = z[i]

            # Fill NaN values
            mask = np.isnan(grid_height)
            idx = np.where(~mask, np.arange(mask.shape[1]), 0)
            np.maximum.accumulate(idx, axis=1, out=idx)
            grid_height[mask] = grid_height[np.nonzero(mask)[0], idx[mask]]

            # Smooth and find local maxima
            grid_height = gaussian_filter(grid_height, sigma=1.0)
            local_max = maximum_filter(grid_height, size=self.window_size)
            tree_tops = (grid_height == local_max) & (grid_height > self.min_height)

            # Get tree positions
            tree_y, tree_x = np.where(tree_tops)
            tree_x_coords = x_grid[tree_x]
            tree_y_coords = y_grid[tree_y]
            tree_heights = grid_height[tree_y, tree_x]

            # Filter close trees
            min_distance = 2.0  # Minimum 2m between trees
            filtered_indices = []

            # Simple filtering: keep first detected tree in each cluster
            for i in range(len(tree_heights)):
                # Calculate distances to all previously accepted trees
                if len(filtered_indices) > 0:
                    prev_x = tree_x_coords[filtered_indices]
                    prev_y = tree_y_coords[filtered_indices]
                    distances = np.sqrt((tree_x_coords[i] - prev_x)**2 +
                                     (tree_y_coords[i] - prev_y)**2)
                    # Only add if not too close to any previous tree
                    if np.min(distances) >= min_distance:
                        filtered_indices.append(i)
                else:
                    # Always add the first tree
                    filtered_indices.append(i)

            # Store results
            self.tree_positions = np.column_stack((
                tree_x_coords[filtered_indices],
                tree_y_coords[filtered_indices]
            ))
            self.tree_heights = tree_heights[filtered_indices]

            print(f"Trees detected: {len(self.tree_heights)}")
            self.trees_detected = True

            return self.tree_positions, self.tree_heights

        except Exception as e:
            print(f"Error in tree detection: {str(e)}")
            raise

    def calculate_dbh(self):
        """Calculate diameter at breast height using allometric equation"""
        try:
            if not self.trees_detected:
                raise ValueError("No trees detected. Run detect_trees first.")

            # Allometric equation: H = 47 * D^0.5
            # Solving for D: D = (H/47)^2
            self.tree_diameters = (self.tree_heights / 47) ** 2
            self.dbh_calculated = True

            print(f"\nDBH Statistics:")
            print(f"Mean DBH: {np.mean(self.tree_diameters):.2f}m")
            print(f"DBH range: {np.min(self.tree_diameters):.2f}m to {np.max(self.tree_diameters):.2f}m")

            return self.tree_diameters

        except Exception as e:
            print(f"Error calculating DBH: {str(e)}")
            return None

    def calculate_basal_area(self, area_ha):
        """Calculate basal area per hectare"""
        try:
            if not self.dbh_calculated:
                self.calculate_dbh()

            # Calculate individual basal areas (π * (d/2)²)
            individual_bas = np.pi * (self.tree_diameters / 2) ** 2

            # Sum and divide by area
            self.basal_area = np.sum(individual_bas) / area_ha
            self.basal_area_calculated = True

            print(f"\nBasal Area: {self.basal_area:.2f} m²/ha")
            return self.basal_area

        except Exception as e:
            print(f"Error calculating basal area: {str(e)}")
            return None

    def plot_detection_results(self, ax, boundary=None):
        """Plot tree detection results"""
        try:
            if not self.trees_detected:
                return

            scatter = ax.scatter(self.tree_positions[:, 0],
                               self.tree_positions[:, 1],
                               c=self.tree_heights,
                               cmap='viridis',
                               s=50,
                               alpha=0.6)

            if boundary is not None:
                boundary.boundary.plot(ax=ax,
                                    color='red',
                                    linewidth=2,
                                    label='Forest Boundary')

            plt.colorbar(scatter, ax=ax, label='Tree Height (m)')
            ax.set_title('Detected Trees', pad=20, weight='bold')
            ax.set_xlabel('X coordinate (m)', labelpad=10)
            ax.set_ylabel('Y coordinate (m)', labelpad=10)
            ax.grid(True, alpha=0.3)
            ax.legend()

        except Exception as e:
            print(f"Error in detection results plot: {str(e)}")

    def plot_diameter_distribution(self, ax):
        """Plot diameter distribution"""
        try:
            if not hasattr(self, 'tree_diameters'):
                return

            # Convert to centimeters for better visualization
            diameters_cm = self.tree_diameters * 100

            # Create histogram
            ax.hist(diameters_cm,
                   bins=30,
                   color='forestgreen',
                   alpha=0.7,
                   edgecolor='black',
                   linewidth=0.5,
                   density=False)  # Set density=False to show actual frequencies

            # Add mean and median lines
            mean_dbh = np.mean(diameters_cm)
            median_dbh = np.median(diameters_cm)
            ax.axvline(mean_dbh, color='red', linestyle='--',
                      label=f'Mean DBH: {mean_dbh:.1f} cm')
            ax.axvline(median_dbh, color='blue', linestyle=':',
                      label=f'Median DBH: {median_dbh:.1f} cm')

            # Set labels and title
            ax.set_title('Tree Diameter Distribution', pad=20, weight='bold')
            ax.set_xlabel('Diameter at Breast Height (cm)', labelpad=10)
            ax.set_ylabel('Frequency', labelpad=10)
            ax.legend(frameon=True, facecolor='white', framealpha=1)
            ax.grid(True, alpha=0.3)

            # Adjust y-axis to show data clearly
            counts, bins = np.histogram(diameters_cm, bins=30)
            ax.set_ylim(0, max(counts) * 1.1)  # Add 10% margin at the top

        except Exception as e:
            print(f"Error in diameter distribution plot: {str(e)}")
            
            
    def validate_detection(self, x, y, z, test_size=0.2, random_state=42):
        """
        Validate tree detection using spatial cross-validation
        """
        try:
            # Create spatial grid for stratification
            x_grid = np.linspace(min(x), max(x), 10)
            y_grid = np.linspace(min(y), max(y), 10)
            grid_indices = []

            # Assign points to grid cells
            for i in range(len(x_grid)-1):
                for j in range(len(y_grid)-1):
                    mask = ((x >= x_grid[i]) & (x < x_grid[i+1]) &
                           (y >= y_grid[j]) & (y < y_grid[j+1]))
                    if np.any(mask):
                        grid_indices.append(np.where(mask)[0])

            # Split grid cells into train/test
            train_cells, test_cells = train_test_split(
                grid_indices,
                test_size=test_size,
                random_state=random_state
            )

            # Create train/test masks
            train_mask = np.zeros(len(x), dtype=bool)
            test_mask = np.zeros(len(x), dtype=bool)

            for cell in train_cells:
                train_mask[cell] = True
            for cell in test_cells:
                test_mask[cell] = True

            # Detect trees on training data
            train_positions, train_heights = self.detect_trees(
                x[train_mask],
                y[train_mask],
                z[train_mask]
            )

            # Detect trees on test data
            test_positions, test_heights = self.detect_trees(
                x[test_mask],
                y[test_mask],
                z[test_mask]
            )

            # Calculate validation metrics
            metrics = self._calculate_detection_metrics(
                train_positions, train_heights,
                test_positions, test_heights
            )

            self.validation_metrics = metrics
            return metrics

        except Exception as e:
            print(f"Error in detection validation: {str(e)}")
            return None

    def _calculate_detection_metrics(self, train_pos, train_heights,
                                   test_pos, test_heights, max_dist=2.0):
        """
        Calculate detection accuracy metrics
        """
        # Calculate distances between train and test trees
        distances = cdist(train_pos, test_pos)

        # Find matching trees
        matches = distances < max_dist
        true_positives = np.sum(matches)
        false_positives = len(train_pos) - true_positives
        false_negatives = len(test_pos) - true_positives

        # Calculate metrics
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * (precision * recall) / (precision + recall)

        # Height error metrics
        matched_indices = np.where(matches)
        height_errors = np.abs(train_heights[matched_indices[0]] -
                             test_heights[matched_indices[1]])

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'mean_height_error': np.mean(height_errors),
            'rmse_height': np.sqrt(np.mean(height_errors**2))
        }