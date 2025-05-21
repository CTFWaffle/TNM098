import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyproj import CRS
from pykml import parser
import os

# Load data
cc_data = pd.read_csv(r'Projekt\data\MC2\cc_data.csv', encoding='cp1252')
gps_data = pd.read_csv(r'Projekt\data\MC2\gps.csv', encoding='cp1252')
loyalty_data = pd.read_csv(r'Projekt\data\MC2\loyalty_data.csv', encoding='cp1252')
car_data = pd.read_csv(r'Projekt\data\MC2\car-assignments.csv', encoding='cp1252')

# Load the .prj file (projection information)
with open(r'Projekt\data\MC2\Geospatial\Abila.prj', 'r') as prj_file:
    prj_content = prj_file.read()
    crs = CRS.from_wkt(prj_content)

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

# Create a GeoDataFrame from the GPS data
gdf_gps = gpd.GeoDataFrame(gps_data, geometry=gpd.points_from_xy(gps_data['long'], gps_data['lat']))

# Create subset of GPS data for a specific id
gps_data_subset = gdf_gps[gdf_gps['id'] == 105].copy()

# Get the bounding box of the data
minx, miny, maxx, maxy = gdf.total_bounds

# Load the base map image
img = plt.imread(r'Projekt\data\MC2\MC2-tourist.jpg')

def save_all_streets_plot(gdf_base, base_map_img, output_path):
    """
    Save the "all streets" plot as an image for reuse.
    """
    if not os.path.exists(output_path):
        minx, miny, maxx, maxy = gdf_base.total_bounds
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(base_map_img, extent=[minx, maxx, miny, maxy])
        gdf_base.plot(ax=ax, color='red', alpha=0.5, edgecolor='k', label='All Streets')
        ax.set_title('All Streets')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.grid()
        plt.savefig(output_path)
        plt.close()
    return output_path

def plot_gps_data_by_day(gps_data, base_map_img, gdf_base, car_id=105, all_streets_img_path=None, movement_threshold=0.00005, max_days=None):
    """
    Create separate plots for each day of GPS data, highlighting spots with minimal movement.
    
    Parameters:
    -----------
    gps_data: GeoDataFrame
        The complete GPS dataset
    base_map_img: array-like
        The base map image
    gdf_base: GeoDataFrame
        The base map GeoDataFrame
    car_id: int
        Car ID to plot (default: 105)
    movement_threshold: float
        Threshold for considering movement as minimal (in coordinate units)
    max_days: int or None
        Maximum number of days to plot. If None, all days will be plotted.
    """
    # Create subset of GPS data for the specified car ID
    gps_data_subset = gps_data[gps_data['id'] == car_id].copy()
    
    if len(gps_data_subset) == 0:
        print(f"No data found for car ID {car_id}")
        return
        
    # Ensure the 'Timestamp' column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(gps_data_subset['Timestamp']):
        gps_data_subset['Timestamp'] = pd.to_datetime(gps_data_subset['Timestamp'])

    # Extract the date from the 'Timestamp' column
    gps_data_subset['Date'] = gps_data_subset['Timestamp'].dt.date
    unique_dates = sorted(gps_data_subset['Date'].unique())  # Sort dates chronologically
    
    # Limit the number of days if max_days is specified
    if max_days is not None:
        unique_dates = unique_dates[:max_days]
    
    # Get car ID for the title
    car_id = gps_data_subset['id'].iloc[0]
    
    # Get the bounding box for consistent display
    minx, miny, maxx, maxy = gdf_base.total_bounds
    
    # Track the previous day's end point
    prev_day_end_x = None
    prev_day_end_y = None
    
    # Create a separate plot for each day
    for i, date in enumerate(unique_dates):
        # Create a new figure with adjusted size to accommodate the sidebar
        fig = plt.figure(figsize=(12, 10))  # Wider figure to fit sidebar
        
        # Create main axis with specific position to leave room for sidebar
        ax = fig.add_axes([0.1, 0.1, 0.75, 0.8])  # [left, bottom, width, height]
        
        # Get data for the current day
        daily_data = gps_data_subset[gps_data_subset['Date'] == date].sort_values('Timestamp').reset_index(drop=True)
        
        # Get day of week
        day_of_week = pd.to_datetime(date).strftime('%A')
        
        # Plot the base map directly
        ax.imshow(base_map_img, extent=[minx, maxx, miny, maxy])
        
        # Plot street data directly
        gdf_base.plot(ax=ax, color='red', alpha=0.5, edgecolor='k', label='All Streets')
        
        # Calculate movement between consecutive points
        daily_data['next_x'] = daily_data['geometry'].x.shift(-1)
        daily_data['next_y'] = daily_data['geometry'].y.shift(-1)
        daily_data['distance'] = np.sqrt(
            (daily_data['geometry'].x - daily_data['next_x'])**2 + 
            (daily_data['geometry'].y - daily_data['next_y'])**2
        )
        daily_data['is_stationary'] = daily_data['distance'] < movement_threshold
        
        # Identify stationary periods
        stationary_periods = []
        start_idx = None
        
        for idx, row in daily_data.iterrows():
            if row['is_stationary'] and start_idx is None:
                start_idx = idx
            elif not row['is_stationary'] and start_idx is not None:
                stationary_periods.append((
                    daily_data.loc[start_idx, 'Timestamp'],
                    daily_data.loc[idx-1, 'Timestamp'],
                    start_idx,
                    idx-1
                ))
                start_idx = None
        
        # Close any open period at the end
        if start_idx is not None:
            stationary_periods.append((
                daily_data.loc[start_idx, 'Timestamp'],
                daily_data.iloc[-1]['Timestamp'],
                start_idx,
                len(daily_data)-1
            ))
        
        # Plot GPS data with simple colors (blue for moving, red for stationary)
        for _, row in daily_data.iterrows():
            x, y = row.geometry.x, row.geometry.y
            color = 'red' if row['is_stationary'] else 'blue'
            marker_size = 8 if row['is_stationary'] else 4
            alpha = 0.9 if row['is_stationary'] else 0.3
            ax.plot(x, y, marker='o', color=color, markersize=marker_size, alpha=alpha)
        
        # Plot start point (first point of the day) with a distinctive marker
        start_x = daily_data.iloc[0].geometry.x
        start_y = daily_data.iloc[0].geometry.y
        ax.plot(start_x, start_y, marker='^', color='green', markersize=14, alpha=1.0)
        
        # Plot end point (last point of the day) with a distinctive marker
        end_x = daily_data.iloc[-1].geometry.x
        end_y = daily_data.iloc[-1].geometry.y
        ax.plot(end_x, end_y, marker='s', color='purple', markersize=10, alpha=1.0)
        
        # Plot previous day's end point with a black X if available
        if prev_day_end_x is not None and prev_day_end_y is not None:
            ax.plot(prev_day_end_x, prev_day_end_y, marker='X', color='black', markersize=14, alpha=1.0)
        
        # Update previous day's end point for the next iteration
        prev_day_end_x = end_x
        prev_day_end_y = end_y
        
        # Add a time progression bar that's clearly outside the main plot
        cbar_ax = fig.add_axes([0.88, 0.1, 0.03, 0.8])  # Positioned further to the right
        cbar_ax.set_title('Time (hours)')
        cbar_ax.set_xticks([])
        
        # Set up the time bar
        min_time = daily_data['Timestamp'].min()
        max_time = daily_data['Timestamp'].max()
        total_minutes = (max_time - min_time).total_seconds() / 60
        
        # Create a blank white background
        cbar_ax.axvspan(0, 1, 0, 1, color='white')
        
        # Add hour markers
        hour_fractions = []
        current_hour = min_time.replace(minute=0, second=0, microsecond=0)
        
        while current_hour <= max_time:
            if current_hour >= min_time:
                fraction = (current_hour - min_time).total_seconds() / (max_time - min_time).total_seconds()
                hour_fractions.append((fraction, current_hour.strftime('%H:%M')))
            current_hour = current_hour + pd.Timedelta(hours=1)
        
        for fraction, label in hour_fractions:
            cbar_ax.axhline(fraction, color='gray', linestyle='--', alpha=0.5)
            cbar_ax.text(0.5, fraction, label, ha='center', va='center')
        
        # Highlight stationary periods on the time bar
        for start, end, start_idx, end_idx in stationary_periods:
            start_pos = (start - min_time).total_seconds() / (max_time - min_time).total_seconds()
            end_pos = (end - min_time).total_seconds() / (max_time - min_time).total_seconds()
            cbar_ax.axhspan(start_pos, end_pos, color='red', alpha=0.3)
            
            # Add text for significant stops (longer than 3 minutes)
            duration = (end - start).total_seconds() / 60
            if duration > 1:
                mid_pos = (start_pos + end_pos) / 2
                time_str = start.strftime('%H:%M')
                duration_str = f"{int(duration)}m"
                
                # Add stop number to track them
                stop_num = len([s for s in stationary_periods 
                              if s[0] <= start and (s[0] - min_time).total_seconds() / 60 > 3])
                
                # Add text to time bar with stop number
                cbar_ax.text(2.0, mid_pos, f"#{stop_num}: {time_str} ({duration_str})", 
                            ha='left', va='center', fontsize=8, color='darkred')
                
                # Calculate the mid-point of this stationary period on the map
                mid_idx = (start_idx + end_idx) // 2
                if mid_idx < len(daily_data):
                    # Get the position of the mid-point
                    mid_x = daily_data.iloc[mid_idx].geometry.x
                    mid_y = daily_data.iloc[mid_idx].geometry.y
                    
                    # Add a numbered label to the map
                    ax.annotate(f"#{stop_num}", 
                              xy=(mid_x, mid_y),
                              xytext=(5, 5),
                              textcoords="offset points",
                              bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.7),
                              fontsize=9,
                              weight='bold')
                
                # Draw a more visible marker at the significant stop
                ax.plot(mid_x, mid_y, marker='o', color='red', 
                       markersize=10, alpha=0.8, zorder=10)

        # Set title with car ID, day count, date, day of week, and stationary period count
        ax.set_title(f'Car {car_id} - Day {i+1}/{len(unique_dates)}: {date} ({day_of_week}) - {len(stationary_periods)} stops')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid()
        
        # Add legend with the new markers
        ax.scatter([], [], color='blue', s=16, label='Moving')
        ax.scatter([], [], color='red', s=64, label='Stationary')
        ax.plot([], [], marker='^', color='green', markersize=14, linestyle='None', label='Start point')
        ax.plot([], [], marker='s', color='purple', markersize=10, linestyle='None', label='End point')
        if i > 0:  # Only add to legend if there's a previous day
            ax.plot([], [], marker='X', color='black', markersize=14, linestyle='None', label='Previous day end')
        ax.legend()
        
        plt.show()

def plot_shared_locations(gdf_gps, time_window='1min', location_precision=5, max_per_plot=20, grid_size=(4, 4)):
    """
    Find and plot locations where multiple IDs were at the same place and time.
    
    Parameters:
    -----------
    gdf_gps: GeoDataFrame
        The GPS dataset with id and location information
    time_window: str
        Time window for grouping (e.g., '1min', '5min')
    location_precision: int
        Decimal places to round coordinates for proximity grouping
    max_per_plot: int
        Maximum number of shared locations to show per plot
    grid_size: tuple
        Number of plots in grid (rows, columns)
    
    Returns:
    --------
    list: Any remaining shared locations not plotted
    """
    # Ensure Timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(gdf_gps['Timestamp']):
        gdf_gps['Timestamp'] = pd.to_datetime(gdf_gps['Timestamp'])

    # Round coordinates and time
    gdf_gps['rounded_x'] = gdf_gps.geometry.x.round(location_precision)
    gdf_gps['rounded_y'] = gdf_gps.geometry.y.round(location_precision)
    gdf_gps['rounded_time'] = gdf_gps['Timestamp'].dt.round(time_window)
    gdf_gps['date'] = gdf_gps['Timestamp'].dt.date

    # Group by date, rounded location, and rounded time
    grouped = gdf_gps.groupby(['date', 'rounded_x', 'rounded_y', 'rounded_time'])

    # Find groups with more than one unique id
    shared = grouped.filter(lambda x: x['id'].nunique() > 1)

    if shared.empty:
        print("No shared locations found.")
        return []

    # Get unique shared location groups
    unique_shared = []
    for _, group in shared.groupby(['date', 'rounded_x', 'rounded_y', 'rounded_time']):
        unique_shared.append({
            'x': group['rounded_x'].iloc[0],
            'y': group['rounded_y'].iloc[0],
            'ids': sorted(group['id'].unique()),
            'time': group['rounded_time'].iloc[0],
            'date': group['date'].iloc[0],
            'group': group
        })
    
    # Calculate how many plots we need
    rows, cols = grid_size
    max_plots = rows * cols
    plots_needed = (len(unique_shared) + max_per_plot - 1) // max_per_plot
    
    if plots_needed == 0:
        return []
    
    # Limit plots to the maximum grid size
    plots_to_create = min(plots_needed, max_plots)
    
    # Create figure with grid of subplots
    if plots_to_create > 1:
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5), squeeze=False)
        axes = axes.flatten()
        # Hide unused subplots
        for i in range(plots_to_create, len(axes)):
            axes[i].axis('off')
    else:
        fig, ax = plt.subplots(figsize=(12, 10))
        axes = [ax]
    
    # Get bounding box for consistent display
    minx, miny, maxx, maxy = gdf.total_bounds
    
    # Plot shared locations in batches
    remaining = []
    total_plotted = 0
    
    for plot_idx in range(plots_to_create):
        if total_plotted >= len(unique_shared):
            break
            
        ax = axes[plot_idx]
        
        # Calculate batch for this plot
        start_idx = plot_idx * max_per_plot
        end_idx = min(start_idx + max_per_plot, len(unique_shared))
        batch = unique_shared[start_idx:end_idx]
        
        # Setup the plot
        ax.imshow(img, extent=[minx, maxx, miny, maxy])
        gdf.plot(ax=ax, color='red', alpha=0.3, edgecolor='k')
        
        # Plot each shared location in this batch
        for item in batch:
            ax.plot(item['x'], item['y'], marker='o', color='orange', markersize=10)
            ids_str = ', '.join(map(str, item['ids']))
            time_str = item['time'].strftime('%Y-%m-%d %H:%M')
            ax.annotate(f"{time_str}\nIDs: {ids_str}", 
                       (item['x'], item['y']), 
                       textcoords="offset points", 
                       xytext=(5,5),
                       bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.7),
                       fontsize=8)
        
        ax.set_title(f"Shared Locations (Batch {plot_idx+1} of {plots_to_create})")
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        
        total_plotted += len(batch)
    
    # Store any remaining items
    if len(unique_shared) > total_plotted:
        remaining = unique_shared[total_plotted:]
    
    plt.tight_layout()
    plt.show()
    
    # Return remaining occurrences
    return remaining

# Find shared locations without plotting
def find_shared_locations(gdf_gps, time_window='1min', location_precision=5, min_ids=2):
    """
    Find locations where multiple IDs were at the same place and time.
    
    Parameters:
    -----------
    gdf_gps: GeoDataFrame
        The GPS dataset with id and location information
    time_window: str
        Time window for grouping (e.g., '1min', '5min')
    location_precision: int
        Decimal places to round coordinates for proximity grouping
    min_ids: int
        Minimum number of unique IDs required to consider a location as shared
        
    Returns:
    --------
    list: All shared locations meeting the criteria
    """
    # Ensure Timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(gdf_gps['Timestamp']):
        gdf_gps['Timestamp'] = pd.to_datetime(gdf_gps['Timestamp'])

    # Round coordinates and time
    gdf_gps['rounded_x'] = gdf_gps.geometry.x.round(location_precision)
    gdf_gps['rounded_y'] = gdf_gps.geometry.y.round(location_precision)
    gdf_gps['rounded_time'] = gdf_gps['Timestamp'].dt.round(time_window)
    gdf_gps['date'] = gdf_gps['Timestamp'].dt.date

    # Group by date, rounded location, and rounded time
    grouped = gdf_gps.groupby(['date', 'rounded_x', 'rounded_y', 'rounded_time'])

    # Find groups with more than min_ids unique id
    shared = grouped.filter(lambda x: x['id'].nunique() >= min_ids)

    if shared.empty:
        print("No shared locations found.")
        return []

    # Get unique shared location groups
    unique_shared = []
    for name, group in shared.groupby(['date', 'rounded_x', 'rounded_y', 'rounded_time']):
        ids = sorted(group['id'].unique())
        
        # Get people's names from car assignments
        people = []
        for id in ids:
            person = car_data[car_data['CarID'] == id]
            if not person.empty:
                name = f"{person['FirstName'].iloc[0]} {person['LastName'].iloc[0]}"
                people.append(f"{id} ({name})")
            else:
                people.append(f"{id}")
                
        unique_shared.append({
            'x': group['rounded_x'].iloc[0],
            'y': group['rounded_y'].iloc[0],
            'ids': ids,
            'people': people,
            'time': group['rounded_time'].iloc[0],
            'date': group['date'].iloc[0],
            'count': len(ids)
        })
    
    # Sort by number of IDs (descending) and then by time
    unique_shared.sort(key=lambda x: (-x['count'], x['time']))
    
    return unique_shared

# Replace the plot_shared_locations call with this:
shared_locations = find_shared_locations(gdf_gps, 
                                       time_window='1min', 
                                       location_precision=5, 
                                       min_ids=2)

print(f"Found {len(shared_locations)} shared location instances")

# Print the top occurrences with the most people
print("\nTOP 20 SHARED LOCATIONS BY NUMBER OF PEOPLE:")
print("-" * 80)
for i, loc in enumerate(shared_locations[:20]):
    print(f"{i+1}. Date/Time: {loc['date']} {loc['time'].strftime('%H:%M')}")
    print(f"   Location: ({loc['x']}, {loc['y']})")
    print(f"   People ({loc['count']}): {', '.join(loc['people'])}")
    print(f"   {'=' * 70}")

# Find significant groups (3+ people)
significant_groups = [loc for loc in shared_locations if loc['count'] >= 3]
print(f"\nFound {len(significant_groups)} significant shared locations (3+ people)")

# Save results to CSV for further analysis
results_df = pd.DataFrame([
    {
        'date': loc['date'],
        'time': loc['time'],
        'x': loc['x'], 
        'y': loc['y'],
        'num_people': loc['count'],
        'people_ids': ', '.join(map(str, loc['ids'])),
        'people_names': ', '.join(loc['people'])
    }
    for loc in shared_locations
])

results_df.to_csv(r'Projekt\shared_locations.csv', index=False)
print(f"\nResults saved to shared_locations.csv")

