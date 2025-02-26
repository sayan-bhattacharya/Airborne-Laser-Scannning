# main.py

from src.data_processing import ForestDataProcessor
from src.visualization import ForestVisualizer, create_analysis_figure
from src.tree_detection import TreeDetector
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def setup_plotting_style():
    """Setup consistent plotting style"""
    # Use default style as base
    plt.style.use('default')

    # Set custom parameters for better visualization
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
        'axes.titlepad': 20,
        'figure.autolayout': True,  # Better layout handling
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })

def save_analysis_results(processor, detector, filename='analysis_results.txt'):
    """Save analysis results to file"""
    try:
        with open(filename, 'w') as f:
            f.write("=== Forest Stand Analysis Results ===\n\n")

            # Point cloud statistics
            f.write("Point Cloud Statistics:\n")
            f.write(f"Total points: {len(processor.x):,}\n")
            f.write(f"Ground level: {processor.ground_level:.2f}m ASL\n")
            f.write(f"Point density: {len(processor.x)/processor.area_ha:.0f} points/ha\n\n")

            # Stand characteristics
            f.write("Stand Characteristics:\n")
            f.write(f"Stand area: {processor.area_ha:.2f} ha\n")
            f.write(f"Number of trees: {len(detector.tree_heights):,}\n")
            f.write(f"Trees per hectare: {len(detector.tree_heights)/processor.area_ha:.0f}\n")
            f.write(f"Mean tree height: {np.mean(detector.tree_heights):.2f}m\n")
            f.write(f"Median tree height: {np.median(detector.tree_heights):.2f}m\n")
            f.write(f"Mean DBH: {np.mean(detector.tree_diameters):.2f}m\n")
            f.write(f"Median DBH: {np.median(detector.tree_diameters):.2f}m\n")
            f.write(f"Basal area: {detector.basal_area:.2f} m²/ha\n\n")

            # Height statistics
            f.write("Height Statistics:\n")
            f.write(f"Mean height: {np.mean(processor.z_normalized):.2f}m\n")
            f.write(f"Median height: {np.median(processor.z_normalized):.2f}m\n")
            f.write(f"Standard deviation: {np.std(processor.z_normalized):.2f}m\n")
            f.write(f"Height range: {processor.z_normalized.min():.2f}m to {processor.z_normalized.max():.2f}m\n")

        print(f"Results saved to '{filename}'")
        return True

    except Exception as e:
        print(f"Error saving results: {str(e)}")
        return False

def validate_data(processor, detector):
    """Validate that all required data is present and valid"""
    try:
        required_processor_attrs = ['x', 'y', 'z', 'z_normalized', 'ground_level', 'area_ha']
        required_detector_attrs = ['tree_positions', 'tree_heights', 'tree_diameters', 'basal_area']

        # Validate processor attributes
        for attr in required_processor_attrs:
            if not hasattr(processor, attr):
                raise ValueError(f"Missing required processor attribute: {attr}")
            if getattr(processor, attr) is None:
                raise ValueError(f"Processor attribute is None: {attr}")

            # Validate numeric arrays
            if attr in ['x', 'y', 'z', 'z_normalized']:
                data = getattr(processor, attr)
                if not isinstance(data, np.ndarray):
                    raise ValueError(f"Processor attribute {attr} must be numpy array")
                if len(data) == 0:
                    raise ValueError(f"Processor attribute {attr} is empty")
                if not np.isfinite(data).all():
                    raise ValueError(f"Processor attribute {attr} contains invalid values")

        # Validate detector attributes
        for attr in required_detector_attrs:
            if not hasattr(detector, attr):
                raise ValueError(f"Missing required detector attribute: {attr}")
            if getattr(detector, attr) is None:
                raise ValueError(f"Detector attribute is None: {attr}")

            # Validate numeric arrays
            if attr in ['tree_positions', 'tree_heights', 'tree_diameters']:
                data = getattr(detector, attr)
                if not isinstance(data, np.ndarray):
                    raise ValueError(f"Detector attribute {attr} must be numpy array")
                if len(data) == 0:
                    raise ValueError(f"Detector attribute {attr} is empty")
                if not np.isfinite(data).all():
                    raise ValueError(f"Detector attribute {attr} contains invalid values")

        return True

    except Exception as e:
        print(f"Data validation error: {str(e)}")
        return False

def create_visualization(processor, detector, visualizer):
    """Create all visualizations"""
    try:
        # Create height distribution plot
        fig_height = plt.figure(figsize=(10, 6))
        visualizer.plot_height_distribution(
            processor.z_normalized,
            processor.ground_level,
            title="Forest Height Distribution"
        )
        plt.tight_layout()
        fig_height.savefig('height_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig_height)

        # Create diameter distribution plot
        fig_dbh = plt.figure(figsize=(10, 6))
        if hasattr(detector, 'tree_diameters') and detector.tree_diameters is not None:
            diameters_cm = detector.tree_diameters * 100  # Convert to cm
            plt.hist(diameters_cm, bins=30, color='forestgreen', alpha=0.7, edgecolor='black')
            plt.axvline(np.mean(diameters_cm), color='red', linestyle='--',
                       label=f'Mean: {np.mean(diameters_cm):.1f}cm')
            plt.axvline(np.median(diameters_cm), color='blue', linestyle=':',
                       label=f'Median: {np.median(diameters_cm):.1f}cm')
            plt.title('Tree Diameter Distribution')
            plt.xlabel('Diameter at Breast Height (cm)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
        fig_dbh.savefig('diameter_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig_dbh)

        # Create point cloud plot
        fig_cloud = plt.figure(figsize=(10, 8))
        visualizer.plot_point_cloud_2d(
            processor.x,
            processor.y,
            processor.z_normalized,
            title="Forest Point Cloud Overview"
        )
        plt.tight_layout()
        fig_cloud.savefig('point_cloud_2d.png', dpi=300, bbox_inches='tight')
        plt.close(fig_cloud)

        # Create comprehensive analysis figure
        fig = create_analysis_figure(processor, detector, visualizer)
        if fig:
            fig.savefig('forest_analysis_results.png', dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            print("Visualizations saved successfully:")
            print("- forest_analysis_results.png")
            print("- height_distribution.png")
            print("- diameter_distribution.png")
            print("- point_cloud_2d.png")
            return True
        else:
            print("Warning: Failed to create analysis figure")
            return False

    except Exception as e:
        print(f"Error in visualization creation: {str(e)}")
        return False

def main():
    try:
        # Setup paths
        current_dir = Path.cwd()
        data_dir = current_dir / "data"
        laz_path = data_dir / "als_data.laz"
        gpkg_path = data_dir / "stand_boundary.gpkg"

        print("=== Forest Analysis System ===")
        print(f"Working directory: {current_dir}")

        # Check if data directory exists
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Check if input files exist
        if not laz_path.exists():
            raise FileNotFoundError(f"LAZ file not found: {laz_path}")
        if not gpkg_path.exists():
            raise FileNotFoundError(f"Geopackage file not found: {gpkg_path}")

        # Initialize processor
        processor = ForestDataProcessor(laz_path=laz_path, gpkg_path=gpkg_path)

        # 1. Data Loading and Preparation
        print("\n1. Data Loading and Preparation")
        if not processor.load_data():
            raise RuntimeError("Failed to load data")

        # 2. Data Processing
        print("\n2. Data Processing")
        processor.clean_data()
        processor.normalize_heights()

        # 3. Tree Detection
        print("\n3. Tree Detection and Analysis")
        detector = TreeDetector(window_size=3, min_height=5)
        tree_positions, tree_heights = detector.detect_trees(
            processor.x, processor.y, processor.z_normalized
        )

        # Calculate tree metrics
        detector.calculate_dbh()
        detector.calculate_basal_area(processor.area_ha)

        # Validate data before visualization
        if not validate_data(processor, detector):
            raise ValueError("Data validation failed")

        # Setup plotting style
        setup_plotting_style()

        # 4. Visualization
        print("\n4. Generating Visualizations")
        visualizer = ForestVisualizer()

        # Create individual plots
        plt.figure(figsize=(10, 6))
        visualizer.plot_height_distribution(
            processor.z_normalized,
            processor.ground_level,
            title="Forest Height Distribution"
        )
        plt.tight_layout()
        plt.savefig('height_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        visualizer.plot_point_cloud_2d(
            processor.x,
            processor.y,
            processor.z_normalized,
            title="Forest Point Cloud Overview"
        )
        plt.tight_layout()
        plt.savefig('point_cloud_2d.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Create analysis figures
        if create_analysis_figure(processor, detector, visualizer):
            print("Visualizations saved successfully:")
            print("- point_cloud_analysis.png")
            print("- tree_analysis.png")
            print("- height_distribution.png")
            print("- point_cloud_2d.png")
        else:
            print("Warning: Failed to create analysis figures")

        # 5. Save Results
        if not save_analysis_results(processor, detector):
            raise RuntimeError("Failed to save analysis results")

        print("\nProcessing complete!")

    except FileNotFoundError as e:
        print(f"\nFile not found error: {str(e)}")
        print("Please ensure all required files are in the correct location.")
        sys.exit(1)
    except ValueError as e:
        print(f"\nValidation error: {str(e)}")
        print("Please check the data processing pipeline.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        print("Please check the input data and dependencies.")
        sys.exit(1)
    finally:
        # Clean up any remaining plots
        plt.close('all')

if __name__ == "__main__":
    main()