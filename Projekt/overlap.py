import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
cc_data = pd.read_csv(r'Projekt\data\MC2\cc_data.csv', encoding='cp1252')
gps_data = pd.read_csv(r'Projekt\data\MC2\gps.csv', encoding='cp1252')
car_assignments = pd.read_csv(r'Projekt\data\MC2\car-assignments.csv', encoding='cp1252')

# Function to find shared locations without plotting
def find_shared_locations(gdf_gps, time_window='1min', location_precision=5, min_ids=2):
    """
    Find locations where multiple IDs were at the same place and time.
    
    Parameters:
    -----------
    gdf_gps: DataFrame or GeoDataFrame
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
    if 'geometry' in gdf_gps.columns:
        # For GeoDataFrame
        gdf_gps['rounded_x'] = gdf_gps.geometry.x.round(location_precision)
        gdf_gps['rounded_y'] = gdf_gps.geometry.y.round(location_precision)
    else:
        # For regular DataFrame
        gdf_gps['rounded_x'] = gdf_gps['long'].round(location_precision)
        gdf_gps['rounded_y'] = gdf_gps['lat'].round(location_precision)
        
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
            person = car_assignments[car_assignments['CarID'] == id]
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

# Find all shared locations
shared_locations = find_shared_locations(gps_data, 
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