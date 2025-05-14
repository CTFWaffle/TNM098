import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyproj import CRS
from pykml import parser
from matplotlib.colors import Normalize
from matplotlib import colormaps  # Use the updated colormap API
import matplotlib.animation as animation

cc_data = pd.read_csv(r'Projekt\data\MC2\cc_data.csv', encoding='cp1252')
gps_data = pd.read_csv(r'Projekt\data\MC2\gps.csv', encoding='cp1252')
loyalty_data = pd.read_csv(r'Projekt\data\MC2\loyalty_data.csv', encoding='cp1252')
car_data = pd.read_csv(r'Projekt\data\MC2\car-assignments.csv', encoding='cp1252')
gps_data = gps_data[gps_data['id'] <= 35]
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

# Plot the base map with all geospatial data
img = plt.imread(r'Projekt\data\MC2\MC2-tourist.jpg')

# Plot GPS data over the map
def plot_gps_data_with_base_map(gps_data, base_map_img, gdf_base):
    # Create a GeoDataFrame from the GPS data
    gdf_gps = gpd.GeoDataFrame(gps_data, geometry=gpd.points_from_xy(gps_data['long'], gps_data['lat']))

    # Create subset of GPS data for a specific id
    gps_data_subset = gdf_gps[gdf_gps['id'] == 35].copy()  # Create a copy to avoid the warning
    
    # Ensure the 'Timestamp' column is in datetime format
    gps_data_subset['Timestamp'] = pd.to_datetime(gps_data_subset['Timestamp'])
    
    # Set the coordinate reference system (CRS) to WGS84
    gps_data_subset.crs = "EPSG:4326"
    
    # Get the bounding box of the base map
    minx, miny, maxx, maxy = gdf_base.total_bounds

    # Normalize the data for color interpolation based on the 'Timestamp' column
    norm = Normalize(vmin=gps_data_subset['Timestamp'].min().timestamp(), 
                     vmax=gps_data_subset['Timestamp'].max().timestamp())
    cmap = colormaps['viridis']  # Updated colormap API

    # Plot the base map with geospatial data and GPS data
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(base_map_img, extent=[minx, maxx, miny, maxy])
    gdf_base.plot(ax=ax, color='red', alpha=0.5, edgecolor='k', label='All Streets')

    # Plot GPS data with interpolated colors based on the 'Timestamp' column
    for _, row in gps_data_subset.iterrows():
        color = cmap(norm(row['Timestamp'].timestamp()))  # Interpolate color based on timestamp
        ax.plot(row.geometry.x, row.geometry.y, marker='o', color=color, markersize=5, alpha=0.1) 

    # Add a color bar to explain the interpolation
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Required for ScalarMappable
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.ax.set_yticks([])  # Remove numeric ticks from the color bar
    cbar.set_label('Interpolation', rotation=270, labelpad=15)

    # Annotate the color bar with "Start" and "End"
    cbar.ax.text(1.5, 0, 'Start', ha='center', va='center', transform=cbar.ax.transAxes)
    cbar.ax.text(1.5, 1, 'End', ha='center', va='center', transform=cbar.ax.transAxes)

    # Add title, labels, and legend
    ax.set_title('Combined Geospatial and GPS Data Plot with Interpolated Colors')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.grid()
    plt.show()

def animate_gps_data_with_base_map(gps_data, base_map_img, gdf_base, subset_id=1, distance_threshold=0.0001):
    gdf_gps = gpd.GeoDataFrame(gps_data, geometry=gpd.points_from_xy(gps_data['long'], gps_data['lat']))
    gps_data_subset = gdf_gps[gdf_gps['id'] == subset_id].copy()
    gps_data_subset['Timestamp'] = pd.to_datetime(gps_data_subset['Timestamp'])
    gps_data_subset = gps_data_subset.sort_values('Timestamp').reset_index(drop=True)
    gps_data_subset.crs = "EPSG:4326"

    # Helper: Thin points by distance threshold (in degrees)
    def thin_points(df, threshold):
        if df.empty:
            return df
        keep = [0]
        last_x, last_y = df.geometry.x.iloc[0], df.geometry.y.iloc[0]
        for i in range(1, len(df)):
            x, y = df.geometry.x.iloc[i], df.geometry.y.iloc[i]
            dist = np.sqrt((x - last_x)**2 + (y - last_y)**2)
            if dist > threshold:
                keep.append(i)
                last_x, last_y = x, y
        return df.iloc[keep].reset_index(drop=True)

    gps_data_subset = thin_points(gps_data_subset, distance_threshold)

    minx, miny, maxx, maxy = gdf_base.total_bounds

    norm = Normalize(vmin=gps_data_subset['Timestamp'].min().timestamp(), 
                     vmax=gps_data_subset['Timestamp'].max().timestamp())
    cmap = colormaps['viridis']

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(base_map_img, extent=[minx, maxx, miny, maxy])
    gdf_base.plot(ax=ax, color='red', alpha=0.5, edgecolor='k', label='All Streets')
    ax.set_title('Animated GPS Data')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.grid()

    scat = ax.scatter([], [], c=[], cmap=cmap, norm=norm, s=20, alpha=0.7)

    def init():
        scat.set_offsets(np.empty((0, 2)))
        scat.set_array([])
        return scat,

    def update(frame):
        data = gps_data_subset.iloc[:frame+1]
        coords = np.column_stack((data.geometry.x, data.geometry.y))
        normed_timestamps = [norm(ts.timestamp()) for ts in data['Timestamp']]
        scat.set_offsets(coords)
        scat.set_array(np.array(normed_timestamps))
        return scat,

    ani = animation.FuncAnimation(
        fig, update, frames=len(gps_data_subset), init_func=init,
        interval=0.001, blit=True, repeat=False
    )

    plt.show()

# Call the new function with the required data
plot_gps_data_with_base_map(gps_data, img, gdf)

# Call the animation function
#animate_gps_data_with_base_map(gps_data, img, gdf, distance_threshold=0.0005)