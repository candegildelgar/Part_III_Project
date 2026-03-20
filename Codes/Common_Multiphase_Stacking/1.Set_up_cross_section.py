# Define a specific cross-section between any 2 latlon points to use as input for multiphase stacking 

from geopy.distance import geodesic
from geopy import Point
import numpy as np
import math

def calculate_initial_bearing(start, end):
    """
    Calculate the initial bearing (forward azimuth) between two points.
    """
    lat1, lon1 = math.radians(start.latitude), math.radians(start.longitude)
    lat2, lon2 = math.radians(end.latitude), math.radians(end.longitude)

    delta_lon = lon2 - lon1
    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon))
    initial_bearing = math.atan2(x, y)
    # Convert from radians to degrees
    initial_bearing = math.degrees(initial_bearing)
    # Normalize to 0-360 degrees
    return (initial_bearing + 360) % 360

# Function to generate the great-circle track between start and end points
def generate_gc_track(start_coords, end_coords, distance_interval_km):
    start_point = Point(start_coords)
    end_point = Point(end_coords)
    
    # Calculate total great-circle distance between start and end points
    total_distance = geodesic(start_point, end_point).km
    
    # Calculate the number of intervals
    num_intervals = int(total_distance // distance_interval_km)
    
    # Calculate the initial bearing from start to end point
    bearing = calculate_initial_bearing(start_point, end_point)
    
    # Initialize track points with the start point
    track_points = [start_point]
    
    # Generate intermediate points along the great-circle path
    for i in range(1, num_intervals):
        # Calculate the distance at this step
        step_distance = i * distance_interval_km
        # Determine the location at this distance along the bearing
        interpolated_point = geodesic(kilometers=step_distance).destination(start_point, bearing)
        track_points.append(interpolated_point)
    
    # Add the end point
    track_points.append(end_point)
    
    return track_points

# Define output file
outfile = "/raid2/cg812/Grids/cross-section_EW_E.lonlat"

# Define start and end coordinates
start_coords = (64.875, -17.14)
end_coords = (64.875, -16.14)

# Define distance interval in km
distance_int= 1

# Generate the great-circle track between start and end points
track = generate_gc_track(start_coords, end_coords, distance_int)
grid_lon = np.array([point.longitude for point in track])
grid_lat = np.array([point.latitude for point in track])

# Write the output to a file using a context manager (for better file handling)
with open(outfile, "w") as text_file:
    for lon, lat in zip(grid_lon, grid_lat):
        text_file.write(f"{lon} {lat}\n")  # Use f-string for more readable formatting

print(f"Cross-section coordinates written to {outfile}")
