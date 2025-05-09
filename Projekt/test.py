import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyproj import CRS
from pykml import parser

# Load cc_data
cc_data = pd.read_csv(r'Projekt\data\MC2\cc_data.csv', encoding='cp1252')

# Load the .prj file (projection information)
with open(r'Projekt\data\MC2\Geospatial\Abila.prj', 'r') as prj_file:
    prj_content = prj_file.read()
    crs = CRS.from_wkt(prj_content)
    #print("CRS:", crs)

# Load the .kml file (geospatial data)
kml_file = r'Projekt\data\MC2\Geospatial\Abila.kml'


# Read the KML file using pykml
with open(kml_file, 'r', encoding="cp1252") as f:
   root = parser.parse(f).getroot()
'''   
# Debugging: Print the structure of each Placemark
for place in root.Document.Folder.Placemark:
    print(etree.tostring(place, pretty_print=True).decode())
'''
# Define the KML namespace
namespace = {'kml': 'http://www.opengis.net/kml/2.2'}

places = []
for place in root.Document.Folder.Placemark:
    # Extract data from ExtendedData
    data = {item.get("name"): item.text for item in
            place.ExtendedData.SchemaData.SimpleData}
    
    # Extract the name of the Placemark
    if hasattr(place, 'name'):
        data['name'] = place.name.text

    # Find the LineString element using the namespace
    linestring = place.find('kml:LineString', namespace)
    if linestring is not None:
        # Find the coordinates within the LineString
        coords = linestring.find('kml:coordinates', namespace)
        if coords is not None:
            data["Coordinates"] = coords.text.strip()
            places.append(data)
'''else:
        print(f"Skipping Placemark without LineString: {place.name if hasattr(place, 'name') else 'Unnamed'}")'''

df = pd.DataFrame(places)

# Filter out rows with invalid or empty coordinates
df = df[df['Coordinates'].apply(lambda x: len(x.split()) > 1)]

# Convert raw coordinates to WKT LINESTRING format
def format_linestring(coords):
    # Split the raw coordinates into individual points
    points = coords.split()
    # Convert "x,y" to "x y" for WKT format
    formatted_points = [point.replace(',', ' ') for point in points]
    # Join the points into a WKT LINESTRING
    return f"LINESTRING ({', '.join(formatted_points)})"

# Apply the formatting function to the Coordinates column
df['Coordinates'] = df['Coordinates'].apply(format_linestring)

# Using the CRS and df to create a GeoDataFrame
geometry = gpd.GeoSeries.from_wkt(df['Coordinates'])
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
gdf = gdf.set_index('name')
gdf = gdf.rename_axis('name').reset_index()

# Convert the geometry to the desired CRS (EPSG:4326)
gdf = gdf.to_crs(epsg=4326)

# Get the bounding box of the data
minx, miny, maxx, maxy = gdf.total_bounds

# Plot geometry data onto MC2 map (MC2-tourist.jpg)
img = plt.imread(r'Projekt\data\MC2\MC2-tourist.jpg')
fig, ax = plt.subplots(figsize=(10, 10))

# Use the bounding box of the data as the extent for the image
ax.imshow(img, extent=[minx, maxx, miny, maxy])

# Plot the geospatial data
gdf.plot(ax=ax, color='red', alpha=0.5, edgecolor='k')

plt.title('Geospatial Data from KML')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid()
plt.show()

print(gdf['name'][1])

# Filter the GeoDataFrame for a specific person by name
person_name = "Skaldis"  # Replace with the actual name
person_gdf = gdf[gdf['name'] == person_name]

# Check if the person exists in the data
if not person_gdf.empty:
    # Plot the base map with all geospatial data
    img = plt.imread(r'Projekt\data\MC2\MC2-tourist.jpg')
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, extent=[minx, maxx, miny, maxy])
    gdf.plot(ax=ax, color='red', alpha=0.5, edgecolor='k', label='All Data')

    # Overlay the specific person's geo-data
    person_gdf.plot(ax=ax, color='blue', alpha=0.7, edgecolor='black', label=f'{person_name}')

    # Add title, labels, and legend
    plt.title('Geospatial Data with Highlighted Person')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid()
    plt.show()
else:
    print(f"No geo-data found for {person_name}")