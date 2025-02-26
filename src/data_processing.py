# src/data_processing.py

import pdal
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import json

class ForestDataProcessor:
    def __init__(self, laz_path, gpkg_path):
        self.laz_path = Path(laz_path)
        self.gpkg_path = Path(gpkg_path)
        self.x = None
        self.y = None
        self.z = None
        self.z_normalized = None
        self.ground_level = None
        self.boundary = None
        self.area_ha = None  # Add area attribute
        self.points_gdf = None  # Add GeoDataFrame for points

    def get_boundary_area(self):
        """
        Calculate and store the area of the forest stand in hectares
        """
        try:
            if self.boundary is None:
                raise ValueError("Boundary not loaded. Run load_data first.")

            # Convert area to hectares (from square meters)
            self.area_ha = self.boundary.geometry.area.iloc[0] / 10000
            print(f"Stand area: {self.area_ha:.2f} ha")
            return self.area_ha

        except Exception as e:
            print(f"Error calculating area: {str(e)}")
            return None

    def filter_points_by_boundary(self):
        """
        Filter points to keep only those within the forest stand boundary
        """
        try:
            if self.boundary is None:
                raise ValueError("Boundary not loaded. Run load_data first.")

            # Create GeoDataFrame from points
            points_geometry = [Point(x, y) for x, y in zip(self.x, self.y)]
            self.points_gdf = gpd.GeoDataFrame(
                geometry=points_geometry,
                data={'z': self.z},
                crs=self.boundary.crs
            )

            # Spatial join with boundary
            points_within = gpd.sjoin(
                self.points_gdf,
                self.boundary,
                how='inner',
                predicate='within'
            )

            # Update point arrays
            self.x = points_within.geometry.x.values
            self.y = points_within.geometry.y.values
            self.z = points_within.z.values

            print(f"Points within boundary: {len(self.x):,}")
            return True

        except Exception as e:
            print(f"Error filtering points by boundary: {str(e)}")
            return False

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
            print(f"Metadata points: {arrays[[1,2,3,4]]}")

            # Extract coordinates
            self.x = arrays['X']
            self.y = arrays['Y']
            self.z = arrays['Z']

            # Load boundary
            print(f"\nLoading boundary from: {self.gpkg_path}")
            self.boundary = gpd.read_file(self.gpkg_path)

            # Calculate area
            self.get_boundary_area()

            # Filter points by boundary
            self.filter_points_by_boundary()

            # Print statistics after boundary filtering
            print("\nPoint Cloud Statistics (within boundary):")
            print(f"Number of points: {len(self.x):,}")
            print(f"X range: {self.x.min():.2f}m to {self.x.max():.2f}m")
            print(f"Y range: {self.y.min():.2f}m to {self.y.max():.2f}m")
            print(f"Height range: {self.z.min():.2f}m to {self.z.max():.2f}m")

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
            if len(self.x) == 0:
                raise ValueError("No points to clean. Run load_data first.")

            # Calculate height statistics
            mean_z = np.mean(self.z)
            std_z = np.std(self.z)

            # Remove extreme outliers
            z_scores = np.abs((self.z - mean_z) / std_z)
            valid_points = z_scores < z_threshold

            # Getth percentile)
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

    def normalize_heights(self):
        """
        Normalize heights by subtracting ground level (5th percentile)
        """
        try:
            if len(self.x) == 0:
                raise ValueError("No points to normalize. Run clean_data first.")

            # Calculate ground level as 5th percentile of heights
            self.ground_level = np.percentile(self.z, 5)

            # Normalize heights by subtracting ground level
            self.z_normalized = self.z - self.ground_level

            print(f"\nHeight Normalization Results:")
            print(f"Ground level: {self.ground_level:.2f}m ASL")
            print(f"Height range: {self.z_normalized.min():.2f}m to {self.z_normalized.max():.2f}m")

        except Exception as e:
            print(f"Error during height normalization: {str(e)}")
            raise