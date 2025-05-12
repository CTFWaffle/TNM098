import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from pyproj import CRS
from pykml import parser

# Load the datasets
credit_card_data = pd.read_csv(r'Projekt\data\MC2\cc_data.csv', encoding='cp1252')
loyalty_card_data = pd.read_csv(r'Projekt\data\MC2\loyalty_data.csv', encoding='cp1252')
gps_data = pd.read_csv(r'Projekt\data\MC2\gps.csv', encoding='cp1252')

# Get unique locations from datasets
#unique_locations = credit_card_data['location'].unique()
#unique_locations_loyalty = loyalty_card_data['location'].unique()

# Get unique locations from both datasets
#unique_locations_combined = set(unique_locations).union(set(unique_locations_loyalty))


# Mapping between the coordinates of places and names
places = {
    'Hallowed Grounds': (0.0,0.0), 
    "Guy's Gyros": (0.0,0.0), 
    'Desafio Golf Course': (0.0,0.0),
    'Bean There Done That': (0.0,0.0), 
    "Frydos Autosupply n' More": (0.0,0.0), 
    "Frank's Fuel": (0.0,0.0), 
    'Daily Dealz': (0.0,0.0), 
    "Jack's Magical Beans": (0.0,0.0), 
    'Nationwide Refinery': (0.0,0.0), 
    'Abila Zacharo': (0.0,0.0), 
    'Kronos Pipe and Irrigation': (0.0,0.0), 
    'Carlyle Chemical Inc.': (0.0,0.0), 
    "Octavio's Office Supplies": (0.0,0.0),
    'Hippokampos': (0.0,0.0), 
    'Abila Scrapyard': (0.0,0.0), 
    'General Grocer': (0.0,0.0), 
    'Abila Airport': (0.0,0.0), 
    'Kronos Mart': (0.0,0.0), 
    'Chostus Hotel': (0.0,0.0), 
    'U-Pump': (0.0,0.0), 
    "Brew've Been Served": (0.0,0.0), 
    'Maximum Iron and Steel': (0.0,0.0), 
    'Roberts and Sons': (0.0,0.0), 
    'Coffee Shack': (0.0,0.0), 
    'Stewart and Sons Fabrication': (0.0,0.0), 
    'Ahaggo Museum': (0.0,0.0), 
    'Katerina’s Café': (0.0,0.0), 
    'Gelatogalore': (0.0,0.0), 
    'Kalami Kafenion': (0.0,0.0), 
    'Brewed Awakenings': (0.0,0.0), 
    'Ouzeri Elian': (0.0,0.0), 
    "Albert's Fine Clothing": (0.0,0.0), 
    "Shoppers' Delight": (0.0,0.0), 
    'Coffee Cameleon': (0.0,0.0)
}

# Match timestamps with places by finding closest GPS coordinates
def match_timestamps_with_places(data1, data2, places):
    """
    Match timestamps from data1 with closest timestamps in data2 to get coordinates.
    
    Args:
        data1: DataFrame with transactions (e.g., credit card data)
        data2: DataFrame with GPS coordinates
        places: Dictionary of place names
    """
    matched_data = []
    processed_locations = set()  # Keep track of locations we've already processed
    
    # Convert timestamps to datetime objects for comparison
    data1 = data1.copy()
    data2 = data2.copy()
    data1['timestamp'] = pd.to_datetime(data1['timestamp'])
    data2['Timestamp'] = pd.to_datetime(data2['Timestamp'])
    
    for index, row1 in data1.iterrows():
        timestamp = row1['timestamp']
        location = row1['location']
        
        # Only process this location if we haven't seen it before
        if location in places and location not in processed_locations:
            # Find GPS points close to this timestamp
            time_diffs = abs(data2['Timestamp'] - timestamp)
            closest_idx = time_diffs.idxmin()
            closest_gps = data2.loc[closest_idx]
            
            # Convert numpy float64 to regular Python floats
            coordinates = (float(closest_gps['long']), float(closest_gps['lat']))
            
            matched_data.append((timestamp, location, coordinates))
            processed_locations.add(location)  # Mark this location as processed
    
    return matched_data

# Match timestamps with places in credit card data using GPS data
matched_cc_data = match_timestamps_with_places(credit_card_data, gps_data, places)

# Read in the image and display it
tourist_image = imread(r'Projekt\data\MC2\MC2-tourist.jpg')

# Extract coordinates from matched_cc_data

longitudes = [coord[0] for _, _, coord in matched_cc_data]
latitudes = [coord[1] for _, _, coord in matched_cc_data]
    
# Find min/max values for axis bounds
min_lon = min(longitudes)
max_lon = max(longitudes)
min_lat = min(latitudes)
max_lat = max(latitudes)
    
# Add a small buffer (5% of range) for better visualization
lon_buffer = (max_lon - min_lon) * 0.01
lat_buffer = (max_lat - min_lat) * 0.01  
    
# Create plot with the tourist image
plt.figure(figsize=(10, 8))
plt.imshow(tourist_image, extent=[min_lon-lon_buffer, max_lon+lon_buffer, min_lat-lat_buffer, max_lat+lat_buffer])
    
# Optionally add points for matched locations
for _, location, (lon, lat) in matched_cc_data:
    plt.scatter(lon, lat, c='red', s=50)
    plt.text(lon, lat, location, fontsize=8)
    
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Map with Transaction Locations')
plt.show()
