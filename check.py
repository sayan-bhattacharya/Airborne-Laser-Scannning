# You can test the LAZ file separately first
import laspy
import pylas

# Try reading with pylas
las = pylas.read("data/als_data.laz")
print("File read successfully with pylas")
print(f"Number of points: {len(las.points)}")