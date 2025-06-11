import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyproj import CRS
from pykml import parser

# =========================
# Data Loading
# =========================

# Load credit card, loyalty card, and GPS datasets
credit_card_data = pd.read_csv(r'Projekt\data\MC2\cc_data.csv', encoding='cp1252')
loyalty_card_data = pd.read_csv(r'Projekt\data\MC2\loyalty_data.csv', encoding='cp1252')
gps_data = pd.read_csv(r'Projekt\data\MC2\gps.csv', encoding='cp1252')

# =========================
# Place Name Mapping (Placeholder Coordinates)
# =========================

# Dictionary mapping place names to coordinates (currently all zeros)
places = {
    'Hallowed Grounds': (0.0, 0.0), 
    "Guy's Gyros": (0.0, 0.0), 
    'Desafio Golf Course': (0.0, 0.0),
    'Bean There Done That': (0.0, 0.0), 
    "Frydos Autosupply n' More": (0.0, 0.0), 
    "Frank's Fuel": (0.0, 0.0), 
    'Daily Dealz': (0.0, 0.0), 
    "Jack's Magical Beans": (0.0, 0.0), 
    'Nationwide Refinery': (0.0, 0.0), 
    'Abila Zacharo': (0.0, 0.0), 
    'Kronos Pipe and Irrigation': (0.0, 0.0), 
    'Carlyle Chemical Inc.': (0.0, 0.0), 
    "Octavio's Office Supplies": (0.0, 0.0),
    'Hippokampos': (0.0, 0.0), 
    'Abila Scrapyard': (0.0, 0.0), 
    'General Grocer': (0.0, 0.0), 
    'Abila Airport': (0.0, 0.0), 
    'Kronos Mart': (0.0, 0.0), 
    'Chostus Hotel': (0.0, 0.0), 
    'U-Pump': (0.0, 0.0), 
    "Brew've Been Served": (0.0, 0.0), 
    'Maximum Iron and Steel': (0.0, 0.0), 
    'Roberts and Sons': (0.0, 0.0), 
    'Coffee Shack': (0.0, 0.0), 
    'Stewart and Sons Fabrication': (0.0, 0.0), 
    'Ahaggo Museum': (0.0, 0.0), 
    'Katerina’s Café': (0.0, 0.0), 
    'Gelatogalore': (0.0, 0.0), 
    'Kalami Kafenion': (0.0, 0.0), 
    'Brewed Awakenings': (0.0, 0.0), 
    'Ouzeri Elian': (0.0, 0.0), 
    "Albert's Fine Clothing": (0.0, 0.0), 
    "Shoppers' Delight": (0.0, 0.0), 
    'Coffee Cameleon': (0.0, 0.0)
}

# =========================
# Helper Function: Match Timestamps with Closest GPS Coordinates
# =========================

def match_timestamps_with_places(data1, data2, places):
    """
    For each unique location in data1, find the GPS coordinate from data2
    with the closest timestamp. Returns a list of (timestamp, location, (lon, lat)).
    Only the first occurrence of each location is processed.

    Args:
        data1: DataFrame with transactions (e.g., credit card data)
        data2: DataFrame with GPS coordinates
        places: Dictionary of place names

    Returns:
        List of tuples: (timestamp, location, (longitude, latitude))
    """
    matched_data = []
    processed_locations = set()  # Track processed locations

    # Convert timestamps to datetime for comparison
    data1 = data1.copy()
    data2 = data2.copy()
    data1['timestamp'] = pd.to_datetime(data1['timestamp'])
    data2['Timestamp'] = pd.to_datetime(data2['Timestamp'])

    for index, row1 in data1.iterrows():
        timestamp = row1['timestamp']
        location = row1['location']

        # Only process each location once
        if location in places and location not in processed_locations:
            # Find GPS point closest in time to the transaction
            time_diffs = abs(data2['Timestamp'] - timestamp)
            closest_idx = time_diffs.idxmin()
            closest_gps = data2.loc[closest_idx]

            # Convert to Python floats for plotting
            coordinates = (float(closest_gps['long']), float(closest_gps['lat']))

            matched_data.append((timestamp, location, coordinates))
            processed_locations.add(location)

    return matched_data

# =========================
# Match Credit Card Transactions to GPS Coordinates
# =========================

matched_cc_data = match_timestamps_with_places(credit_card_data, gps_data, places)

# =========================
# Visualization: Plot Transaction Locations on Tourist Map
# =========================

# Load the tourist map image
tourist_image = plt.imread(r'Projekt\data\MC2\MC2-tourist.jpg')

# Extract longitude and latitude from matched data
longitudes = [coord[0] for _, _, coord in matched_cc_data]
latitudes = [coord[1] for _, _, coord in matched_cc_data]

# Determine axis bounds with a small buffer for better visualization
min_lon = min(longitudes)
max_lon = max(longitudes)
min_lat = min(latitudes)
max_lat = max(latitudes)
lon_buffer = (max_lon - min_lon) * 0.01
lat_buffer = (max_lat - min_lat) * 0.01

# Create the plot
plt.figure(figsize=(10, 8))
plt.imshow(
    tourist_image,
    extent=[min_lon - lon_buffer, max_lon + lon_buffer, min_lat - lat_buffer, max_lat + lat_buffer]
)

# Plot each matched location as a red dot and label it
for _, location, (lon, lat) in matched_cc_data:
    plt.scatter(lon, lat, c='red', s=50)
    plt.text(lon, lat, location, fontsize=8)

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Map with Transaction Locations')
plt.show()
