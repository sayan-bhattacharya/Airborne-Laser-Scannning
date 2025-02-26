import laspy
import geopandas as gpd
import json
import pdal

def inspect_laz_metadata(laz_path):
    """Inspect LAZ file metadata using both laspy and pdal"""
    print("\n=== LAZ File Metadata ===")

    # Using laspy for basic header information
    with laspy.open(laz_path) as laz_file:
        header = laz_file.header
        print("\nHeader Information:")
        print(f"Point format:          {header.point_format}")
        print(f"Number of points:      {header.point_count:,}")
        print(f"Number of VLRs:        {len(header.vlrs)}")
        print(f"Scales:                X:{header.x_scale}, Y:{header.y_scale}, Z:{header.z_scale}")
        print(f"Offsets:               X:{header.x_offset}, Y:{header.y_offset}, Z:{header.z_offset}")
        print(f"Min values:            X:{header.x_min}, Y:{header.y_min}, Z:{header.z_min}")
        print(f"Max values:            X:{header.x_max}, Y:{header.y_max}, Z:{header.z_max}")

    # Using PDAL for detailed point dimensions
    try:
        pipeline = pdal.Pipeline(json.dumps({
            "pipeline": [
                {
                    "type": "readers.las",
                    "filename": laz_path
                }
            ]
        }))
        pipeline.execute()
        metadata = pipeline.metadata

        # Access schema directly from metadata dictionary
        schema = metadata['metadata']['schema']['dimensions']

        print("\nAvailable Point Dimensions:")
        for dim in schema:
            print(f"- {dim['name']}: {dim['type']}")
    except Exception as e:
        print(f"\nError reading PDAL metadata: {str(e)}")

    # Print VLR information if available
    print("\nVLR Information:")
    with laspy.open(laz_path) as laz_file:
        for vlr in laz_file.header.vlrs:
            print(f"- {vlr.user_id}: {vlr.description}")

def inspect_gpkg_metadata(gpkg_path):
    """Inspect GPKG file metadata"""
    try:
        print("\n=== GPKG File Metadata ===")
        gdf = gpd.read_file(gpkg_path)

        print("\nDataset Information:")
        print(f"Number of features:    {len(gdf)}")
        print(f"Geometry type:         {gdf.geometry.type[0]}")
        print(f"CRS:                   {gdf.crs}")
        print(f"Total bounds:          {gdf.total_bounds}")

        print("\nAvailable Columns:")
        for col in gdf.columns:
            print(f"- {col}: {gdf[col].dtype}")

        # Print first feature's properties
        print("\nSample Feature Properties:")
        sample_feature = gdf.iloc[0].to_dict()
        feature_dict = {k: str(v) for k, v in sample_feature.items() if k != 'geometry'}
        print(json.dumps(feature_dict, indent=2))
    except Exception as e:
        print(f"\nError reading GPKG file: {str(e)}")

if __name__ == "__main__":
    try:
        # Inspect LAZ file
        laz_path = "data/als_data.laz"
        inspect_laz_metadata(laz_path)

        # Inspect GPKG file
        gpkg_path = "data/stand_boundary.gpkg"
        inspect_gpkg_metadata(gpkg_path)
    except Exception as e:
        print(f"\nMain execution error: {str(e)}")

