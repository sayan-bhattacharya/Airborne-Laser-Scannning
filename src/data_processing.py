# src/data_processing.py

import pdal
import numpy as np
import geopandas as gpd
from pathlib import Path
import json

class ForestDataProcessor:
    def __init__(self, laz_path, gpkg_path):
        self.laz_path = Path(laz_path)
        self.gpkg_path = Path(gpkg_path)
        self.x = None
        self.y = None
        self.z = None

    def load_data(self):
        """
        Load both LAZ and Geopackage data using PDAL
        """
        try:
            # Verify file existence
            if not self.laz_path.exists():
                raise FileNotFoundError(f"LAZ file not found at {self.laz_path}")
            if not self.gpkg_path.exists():
                raise FileNotFoundError(f"GPKG file not found at {self.gpkg_path}")

            print(f"Loading LAZ file from: {self.laz_path}")

            # Define PDAL pipeline
            pipeline_json = {
                "pipeline": [
                    str(self.laz_path),
                    {
                        "type": "filters.range",
                        "limits": "Classification![7:7]"  # Remove noise points
                    }
                ]
            }

            # Execute pipeline
            pipeline = pdal.Pipeline(json.dumps(pipeline_json))
            pipeline.execute()

            # Get array of points
            arrays = pipeline.arrays[0]

            # Extract coordinates
            self.x = arrays['X']
            self.y = arrays['Y']
            self.z = arrays['Z']

            # Basic statistics
            print("\nPoint Cloud Statistics:")
            print(f"Number of points: {len(self.x):,}")
            print(f"X range: {self.x.min():.2f}m to {self.x.max():.2f}m")
            print(f"Y range: {self.y.min():.2f}m to {self.y.max():.2f}m")
            print(f"Height range: {self.z.min():.2f}m to {self.z.max():.2f}m")

            # Load boundary
            print(f"\nLoading boundary from: {self.gpkg_path}")
            self.boundary = gpd.read_file(self.gpkg_path)
            print("Boundary data loaded successfully!")

            return True

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            print("Please ensure all files are in the correct location and format.")
            return False

    def clean_data(self, z_threshold=3):
        """
        Clean point cloud data
        """
        try:
            # Calculate height statistics
            mean_z = np.mean(self.z)
            std_z = np.std(self.z)

            # Remove extreme outliers
            z_scores = np.abs((self.z - mean_z) / std_z)
            valid_points = z_scores < z_threshold

            # Get ground level estimate (5th percentile)
            ground_level = np.percentile(self.z, 5)

            # Remove points too close to ground
            valid_points &= (self.z > ground_level + 0.5)  # 0.5m above ground level

            # Apply filters
            self.x = self.x[valid_points]
            self.y = self.y[valid_points]
            self.z = self.z[valid_points]

            print(f"\nData Cleaning Results:")
            print(f"Points after cleaning: {len(self.x):,}")
            print(f"Ground level estimate: {ground_level:.2f}m")

        except Exception as e:
            print(f"Error during data cleaning: {str(e)}")

    def get_boundary_area(self):
        """
        Calculate the area of the forest stand in hectares
        """
        try:
            # Convert area to hectares (from square meters)
            area_ha = self.boundary.geometry.area.iloc[0] / 10000
            return area_ha
        except Exception as e:
            print(f"Error calculating area: {str(e)}")
            return None


