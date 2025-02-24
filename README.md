# Forest Pulse Analysis

A Python-based tool for analyzing and visualizing LiDAR forest data.

## Environment Setup

### Prerequisites

- Anaconda or Miniconda
- PDAL (Point Data Abstraction Library)
- Python 3.12+

### Step 1: Create and Activate Conda Environment

```bash
# Create a new conda environment named 'forest' with Python 3.12
conda create -n forest python=3.12

# Activate the environment
conda activate forest
```

### Step 2: Install Required Packages

```bash
# Install core dependencies using conda
conda install -c conda-forge pdal python-pdal
conda install numpy matplotlib seaborn

# Install additional dependencies using pip
pip install laspy[laszip]  # For LAZ file processing
```

### Step 3: Verify Installation

```bash
# Verify PDAL installation
pdal --version

# Verify Python environment
python -c "import pdal; import laspy; import seaborn; print('All dependencies installed successfully!')"
```

## Project Structure

```
forest_pulse_analysis/
├── data/
│   ├── als_data.laz          # LiDAR point cloud data
│   └── stand_boundary.gpkg    # Forest stand boundary data
├── src/
│   ├── data_processing.py
│   └── visualization.py
└── main.py
```

## Data Requirements

Place your input data files in the `data/` directory:

- LiDAR point cloud data (`.laz` format)
- Forest stand boundary (`.gpkg` format)

## Running the Analysis

1. Ensure you're in the conda environment:

```bash
conda activate forest
```

2. Navigate to the project directory:

```bash
cd path/to/forest_pulse_analysis
```

3. Run the main script:

```bash
python main.py
```

## Expected Output

The script will:

1. Load and process the LiDAR point cloud data
2. Display point cloud statistics
3. Generate visualizations including:
   - 2D top-down view of the point cloud
   - Height distribution histogram
   - 3D point cloud sample visualization

## Troubleshooting

If you encounter the error "seaborn is not a valid package style", ensure you're using the correct seaborn version:

```bash
conda install seaborn=0.13.2
```

For any PDAL-related issues, verify that PDAL is properly installed in your system:

```bash
conda install -c conda-forge pdal python-pdal
```

## Dependencies

- numpy
- matplotlib
- seaborn
- pdal
- laspy[laszip]
- python-pdal

## License

[Specify your license here]
