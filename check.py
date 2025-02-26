import laspy

# Path to your .laz file
file_path = "/Volumes/T7 Shield/Codebase/HAWK_assignment/forest_pulse_analysis/data/als_data.laz"

# Open the LAZ file
with laspy.open(file_path) as las_file:
    print(las_file.header)
    print(las_file.header)
  