import pandas as pd
from fastkml import kml
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
from pyproj import CRS
from pykml import parser

# Load the datasets
credit_card_data = pd.read_csv(r'Projekt\data\MC2\cc_data.csv', encoding='cp1252')
loyalty_card_data = pd.read_csv(r'Projekt\data\MC2\loyalty_data.csv', encoding='cp1252')
# Load the .kml file (geospatial data)
kml_file = r'Projekt\data\MC2\Geospatial\Abila.kml'

# Read the KML file using pykml
with open(kml_file, 'r', encoding="cp1252") as f:
   root = parser.parse(f).getroot()
namespace = {'kml': 'http://www.opengis.net/kml/2.2'}

places = []
location_coords = {}  # Dictionary to store location name -> coordinates mapping

for place in root.Document.Folder.Placemark:
    # Extract data from ExtendedData
    data = {item.get("name"): item.text for item in
            place.ExtendedData.SchemaData.SimpleData}
    
    # Find the LineString element using the namespace
    linestring = place.find('kml:LineString', namespace)
    if linestring is not None:
        # Find the coordinates within the LineString
        coords = linestring.find('kml:coordinates', namespace)
        if coords is not None:
            data["Coordinates"] = coords.text.strip()
            places.append(data)
            
            # If this place has a name, store its coordinates (use average of all points)
            if "name" in data:
                location_name = data["name"]
                coord_pairs = coords.text.strip().split()
                lons = []
                lats = []
                for pair in coord_pairs:
                    parts = pair.split(',')
                    if len(parts) >= 2:
                        lons.append(float(parts[0]))
                        lats.append(float(parts[1]))
                
                if lons and lats:  # If we have coordinates
                    # Use average point as the location's coordinates
                    location_coords[location_name] = {
                        'lon': sum(lons) / len(lons),
                        'lat': sum(lats) / len(lats)
                    }

df = pd.DataFrame(places)

# Extract coordinates from df for map boundaries
all_lons = []
all_lats = []

# Parse coordinates from the df dataframe
for coord_str in df['Coordinates']:
    coord_pairs = coord_str.strip().split()
    for pair in coord_pairs:
        parts = pair.split(',')
        if len(parts) >= 2:
            lon, lat = float(parts[0]), float(parts[1])
            all_lons.append(lon)
            all_lats.append(lat)

# Add coordinates from card transactions based on location names
# For credit card data
credit_card_coords = {'lon': [], 'lat': []}
for location in credit_card_data['location']:
    if location in location_coords:
        credit_card_coords['lon'].append(location_coords[location]['lon'])
        credit_card_coords['lat'].append(location_coords[location]['lat'])

# For loyalty card data
loyalty_card_coords = {'lon': [], 'lat': []}
for location in loyalty_card_data['location']:
    if location in location_coords:
        loyalty_card_coords['lon'].append(location_coords[location]['lon'])
        loyalty_card_coords['lat'].append(location_coords[location]['lat'])

# Add transaction coordinates to get complete bounds
all_lons.extend(credit_card_coords['lon'])
all_lats.extend(credit_card_coords['lat'])
all_lons.extend(loyalty_card_coords['lon'])
all_lats.extend(loyalty_card_coords['lat'])

# Add padding around the boundaries (adjust as needed)
padding = 0.001  # Degrees of padding

# Calculate the min/max values
if all_lons and all_lats:  # Check if lists are not empty
    min_lon = min(all_lons) - padding
    max_lon = max(all_lons) + padding
    min_lat = min(all_lats) - padding
    max_lat = max(all_lats) + padding
else:
    # Default values if no coordinates found
    min_lon, max_lon = -1, 1
    min_lat, max_lat = -1, 1

print(f"Map boundaries: {min_lon}, {max_lon}, {min_lat}, {max_lat}")

# Plot data on the tourist map
tourist_image = imread(r'Projekt\data\MC2\MC2-tourist.jpg')

plt.figure(figsize=(12, 10))
plt.imshow(tourist_image, extent=[min_lon, max_lon, min_lat, max_lat])

# Plot credit card transactions
plt.scatter(credit_card_coords['lon'], credit_card_coords['lat'], 
           c='red', alpha=0.5, s=20, label='Credit Card')

# Plot loyalty card transactions
plt.scatter(loyalty_card_coords['lon'], loyalty_card_coords['lat'], 
           c='blue', alpha=0.5, s=20, label='Loyalty Card')


print(f"Number of credit card points plotted: {len(credit_card_coords['lon'])}")
print(f"Number of loyalty card points plotted: {len(loyalty_card_coords['lon'])}")


plt.legend()
plt.title('Transaction Locations in Abila')
plt.show()
