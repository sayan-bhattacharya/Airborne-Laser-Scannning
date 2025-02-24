# main.py

from src.data_processing import ForestDataProcessor
from src.visualization import ForestVisualizer
import sys
from pathlib import Path
import laspy

def load_laz_file(file_path):
    try:
        # Try lazrs backend first, fall back to laszip if not available
        try:
            las_data = laspy.read(file_path, laz_backend=laspy.LazBackend.Lazrs)
            print("Using Lazrs backend")
            return las_data
        except (ImportError, ValueError):
            # If lazrs fails, try laszip
            try:
                las_data = laspy.read(file_path, laz_backend=laspy.LazBackend.Laszip)
                print("Using Laszip backend")
                return las_data
            except (ImportError, ValueError):
                # If both fail, try automatic backend selection
                las_data = laspy.read(file_path)
                print("Using automatic backend selection")
                return las_data
    except Exception as e:
        print(f"Error loading LAZ file: {e}")
        return None

def main():
    try:
        # Get absolute paths
        current_dir = Path(__file__).parent
        laz_path = current_dir / "data" / "als_data.laz"
        gpkg_path = current_dir / "data" / "stand_boundary.gpkg"

        print(f"Working directory: {current_dir}")
        print(f"Looking for LAZ file at: {laz_path}")
        print(f"Looking for GPKG file at: {gpkg_path}")

        # Initialize processors
        processor = ForestDataProcessor(
            laz_path=laz_path,
            gpkg_path=gpkg_path
        )

        # Load data
        if not processor.load_data():
            print("Failed to load data. Exiting...")
            return

        print("\nStarting data processing...")

        # Clean data
        processor.clean_data()

        # Initialize visualizer
        visualizer = ForestVisualizer()

        print("\nGenerating visualizations...")

        # Create visualizations
        visualizer.plot_height_distribution(processor.z)
        visualizer.plot_point_cloud_2d(processor.x, processor.y, processor.z)
        visualizer.plot_3d_sample(processor.x, processor.y, processor.z)

        print("\nProcessing complete!")

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please ensure all files are in the correct location and all dependencies are installed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
