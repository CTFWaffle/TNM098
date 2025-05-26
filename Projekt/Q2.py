import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from sklearn.cluster import MiniBatchKMeans
import plotly.graph_objs as go
import plotly.express as px

# Define the data directory
data_dir = "Projekt/data/MC2/"

# Load credit card data
cc_data = pd.read_csv(os.path.join(data_dir, "cc_data.csv"), parse_dates=['timestamp'], encoding='cp1252')

# Load loyalty card data
loyalty_data = pd.read_csv(os.path.join(data_dir, "loyalty_data.csv"), parse_dates=['timestamp'], encoding='cp1252')

# Load GPS data
gps_data = pd.read_csv(os.path.join(data_dir, "gps.csv"), parse_dates=['Timestamp'], encoding='cp1252')

# Load car assignments
car_assignments = pd.read_csv(os.path.join(data_dir, "car-assignments.csv"), encoding='cp1252')

def preprocess_gps_data(gps_df):
    """Preprocess GPS data for easier matching"""
    # Rename columns for consistency
    gps_df = gps_df.rename(columns={
        'Timestamp': 'timestamp',
        'id': 'car_id',
        'lat': 'latitude',
        'long': 'longitude'
    })
    return gps_df

def get_nearest_location(lat, lon, location_dict):
    """Find the nearest known location given coordinates"""
    min_dist = float('inf')
    nearest_loc = None
    
    for loc_name, loc_coords in location_dict.items():
        loc_lat, loc_lon = loc_coords
        dist = ((lat - loc_lat)**2 + (lon - loc_lon)**2)**0.5
        if dist < min_dist:
            min_dist = dist
            nearest_loc = loc_name
    
    return nearest_loc, min_dist

def match_transaction_to_gps(transaction, gps_df, time_window=600):  # Increased from 300 to 600 seconds
    """Match a transaction to GPS records within a time window (in seconds)"""
    transaction_time = transaction['timestamp']
    # Filter GPS records within time window
    time_lower = transaction_time - pd.Timedelta(seconds=time_window)
    time_upper = transaction_time + pd.Timedelta(seconds=time_window)
    
    nearby_gps = gps_df[(gps_df['timestamp'] >= time_lower) & 
                        (gps_df['timestamp'] <= time_upper)]
    
    return nearby_gps

def link_transaction_to_employee(transaction, nearby_gps, car_assignments_df):
    """Link a transaction to an employee through car assignments"""
    if nearby_gps.empty:
        return None, None
    
    # Get unique car IDs from nearby GPS records
    car_ids = nearby_gps['car_id'].unique()
    
    # For each car, check if it's assigned to an employee
    for car_id in car_ids:
        employee = car_assignments_df[car_assignments_df['CarID'] == car_id]
        if not employee.empty:
            return employee['LastName'].iloc[0], employee['FirstName'].iloc[0]
    
    return None, None

def identify_key_locations():
    """Use clustering to identify key locations from GPS data"""
    print("Identifying key locations with MiniBatchKMeans clustering...")
    
    # Extract coordinates
    coords = gps_data[['lat', 'long']].values
    print(f"Extracted {len(coords)} GPS points for clustering")
    
    # Sample data if it's too large to process
    max_points = 700000  # Set a reasonable maximum
    if len(coords) > max_points:
        print(f"Dataset too large, sampling {max_points} points from {len(coords)} total points")
        # Use random sampling without replacement
        sample_idx = np.random.choice(len(coords), max_points, replace=False)
        coords = coords[sample_idx]
        print(f"Sampled down to {len(coords)} points")
    
    # Filter out potential outliers before clustering
    def filter_gps_outliers(coords):
        # Remove points with extreme lat/long values
        lat_q1, lat_q3 = np.percentile(coords[:, 0], [1, 99])
        lon_q1, lon_q3 = np.percentile(coords[:, 1], [1, 99])
        
        mask = (
            (coords[:, 0] >= lat_q1) & 
            (coords[:, 0] <= lat_q3) & 
            (coords[:, 1] >= lon_q1) & 
            (coords[:, 1] <= lon_q3)
        )
        
        return coords[mask]

    # Apply filtering before clustering
    
    #coords = filter_gps_outliers(coords)
    #print(f"After filtering outliers: {len(coords)} points")
    
    filtered_coords = coords  # <--- Save this for plotting
    
    # Use MiniBatchKMeans for clustering
    n_clusters = 34  
    clustering = MiniBatchKMeans(
        n_clusters=n_clusters, 
        batch_size=10000,
        random_state=42
    ).fit(filtered_coords)
    
    labels = clustering.labels_
    n_clusters = len(set(labels))
    print(f"Found {n_clusters} distinct location clusters")
    
    # Get cluster centers
    clusters = {}
    for i, label in enumerate(labels):
        if label == -1:  # Skip noise points
            continue
        
        if label not in clusters:
            clusters[label] = []
        
        clusters[label].append(coords[i])
    
    # Calculate cluster centers
    locations = {}
    for i, points in clusters.items():
        avg_lat = sum(p[0] for p in points) / len(points)
        avg_long = sum(p[1] for p in points) / len(points)
        
        # Try to match with named locations from transaction data if possible
        locations[f"Location {i+1}"] = (avg_lat, avg_long)
    
    # Try to label clusters with actual location names from transaction data
    location_mapping = map_clusters_to_names(locations, cc_data, loyalty_data)
    
    # Replace generic names with actual location names where possible
    final_locations = {}
    for loc_id, coords in locations.items():
        if loc_id in location_mapping:
            final_locations[location_mapping[loc_id]] = coords
        else:
            final_locations[loc_id] = coords
    
    # Return locations, coords, labels, and the clustering model
    return final_locations, filtered_coords, labels, clustering

def map_clusters_to_names(cluster_locations, cc_data, loyalty_data):
    """Map each cluster to at most one location name and vice versa (greedy matching)."""
    processed_gps = preprocess_gps_data(gps_data)
    location_mapping = {}

    # Get all unique location names from transaction data
    all_locations = set(cc_data['location'].unique()) | set(loyalty_data['location'].unique())

    # For each named location, find average GPS position
    location_avg_coords = {}
    for location_name in all_locations:
        all_gps_near_location = []

        cc_at_location = cc_data[cc_data['location'] == location_name]
        loyalty_at_location = loyalty_data[loyalty_data['location'] == location_name]

        for _, transaction in cc_at_location.iterrows():
            nearby_gps = match_transaction_to_gps(transaction, processed_gps)
            if not nearby_gps.empty:
                all_gps_near_location.extend(nearby_gps[['latitude', 'longitude']].values.tolist())
        for _, transaction in loyalty_at_location.iterrows():
            nearby_gps = match_transaction_to_gps(transaction, processed_gps)
            if not nearby_gps.empty:
                all_gps_near_location.extend(nearby_gps[['latitude', 'longitude']].values.tolist())

        if all_gps_near_location:
            avg_lat = sum(point[0] for point in all_gps_near_location) / len(all_gps_near_location)
            avg_long = sum(point[1] for point in all_gps_near_location) / len(all_gps_near_location)
            location_avg_coords[location_name] = (avg_lat, avg_long)

    # Compute all distances between clusters and named locations
    pairs = []
    for loc_name, (loc_lat, loc_long) in location_avg_coords.items():
        for cluster_id, (cluster_lat, cluster_long) in cluster_locations.items():
            dist = ((loc_lat - cluster_lat) ** 2 + (loc_long - cluster_long) ** 2) ** 0.5
            pairs.append((dist, loc_name, cluster_id))

    # Sort all pairs by distance
    pairs.sort()

    # Greedily assign closest pairs, ensuring one-to-one mapping
    used_locations = set()
    used_clusters = set()
    for dist, loc_name, cluster_id in pairs:
        if dist < 0.05 and loc_name not in used_locations and cluster_id not in used_clusters:
            location_mapping[cluster_id] = loc_name
            used_locations.add(loc_name)
            used_clusters.add(cluster_id)

    print(f"Successfully mapped {len(location_mapping)} clusters to named locations (one-to-one)")
    return location_mapping

def analyze_transactions(known_locations):
    """Main analysis function to process all transactions"""
    # Preprocess GPS data
    processed_gps = preprocess_gps_data(gps_data)
    
    # Create dictionaries to store results
    employee_profiles = {}
    
    # Process credit card transactions
    print("Processing credit card transactions...")
    for idx, transaction in cc_data.iterrows():
        # Find nearby GPS records
        nearby_gps = match_transaction_to_gps(transaction, processed_gps)
        
        # Link to employee
        last_name, first_name = link_transaction_to_employee(transaction, nearby_gps, car_assignments)
        
        if last_name and first_name:
            employee_id = f"{first_name} {last_name}"
            
            # Create employee profile if it doesn't exist
            if employee_id not in employee_profiles:
                employee_profiles[employee_id] = {
                    'credit_transactions': [],
                    'loyalty_transactions': [],
                    'locations_visited': set(),
                    'total_spent': 0,
                    'movement_patterns': []
                }
            
            # Add transaction to employee profile
            employee_profiles[employee_id]['credit_transactions'].append({
                'timestamp': transaction['timestamp'],
                'location': transaction['location'],
                'price': transaction['price'],
                'car_id': nearby_gps['car_id'].iloc[0] if not nearby_gps.empty else None
            })
            
            employee_profiles[employee_id]['total_spent'] += transaction['price']
            employee_profiles[employee_id]['locations_visited'].add(transaction['location'])
    
    # Process loyalty card transactions (similar approach)
    print("Processing loyalty card transactions...")
    for idx, transaction in loyalty_data.iterrows():
        nearby_gps = match_transaction_to_gps(transaction, processed_gps)
        last_name, first_name = link_transaction_to_employee(transaction, nearby_gps, car_assignments)
        
        if last_name and first_name:
            employee_id = f"{first_name} {last_name}"
            
            if employee_id not in employee_profiles:
                employee_profiles[employee_id] = {
                    'credit_transactions': [],
                    'loyalty_transactions': [],
                    'locations_visited': set(),
                    'total_spent': 0,
                    'movement_patterns': []
                }
            
            employee_profiles[employee_id]['loyalty_transactions'].append({
                'timestamp': transaction['timestamp'],
                'location': transaction['location']
            })
            
            employee_profiles[employee_id]['locations_visited'].add(transaction['location'])
    
    # Add GPS movement patterns to profiles
    print("Adding movement patterns...")
    for employee, car_info in car_assignments.iterrows():
        employee_id = f"{car_info['FirstName']} {car_info['LastName']}"
        car_id = car_info['CarID']
        
        # Skip if employee hasn't been linked to any transactions
        if employee_id not in employee_profiles:
            employee_profiles[employee_id] = {
                'credit_transactions': [],
                'loyalty_transactions': [],
                'locations_visited': set(),
                'total_spent': 0,
                'movement_patterns': []
            }
        
        # Get all GPS records for this car
        car_gps = processed_gps[processed_gps['car_id'] == car_id].sort_values('timestamp')
        
        # Process movement patterns
        for idx, gps_record in car_gps.iterrows():
            nearest_loc, dist = get_nearest_location(gps_record['latitude'], gps_record['longitude'], known_locations)
            
            employee_profiles[employee_id]['movement_patterns'].append({
                'timestamp': gps_record['timestamp'],
                'latitude': gps_record['latitude'],
                'longitude': gps_record['longitude'],
                'nearest_location': nearest_loc,
                'distance_to_location': dist
            })
            
            if nearest_loc:
                employee_profiles[employee_id]['locations_visited'].add(nearest_loc)
    
    return employee_profiles

# Generate locations using clustering
known_locations, clustering_coords, clustering_labels, clustering_model = identify_key_locations()

# Now run the analysis with the identified locations
employee_profiles = analyze_transactions(known_locations)

def analyze_results(profiles):
    """Analyze the results to identify patterns and anomalies and save to CSV"""
    # Find popular locations
    location_counts = {}
    for emp_id, profile in profiles.items():
        for location in profile['locations_visited']:
            if location not in location_counts:
                location_counts[location] = 0
            location_counts[location] += 1
    
    # Popular locations
    popular_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)
    print("Most popular locations:")
    for loc, count in popular_locations[:10]:
        print(f"  {loc}: {count} visits")
    
    # Create DataFrame for all location visits by employee
    location_visits_data = []
    for emp_id, profile in profiles.items():
        for location in profile['locations_visited']:
            # Count visits to this specific location by this employee
            visit_count = 0
            for movement in profile['movement_patterns']:
                if movement['nearest_location'] == location:
                    visit_count += 1
            
            location_visits_data.append({
                'employee': emp_id,
                'location': location,
                'visit_count': visit_count,
                'total_spent': sum(t['price'] for t in profile['credit_transactions'] 
                                 if t['location'] == location)
            })
    
    # Convert to DataFrame and save to CSV
    location_visits_df = pd.DataFrame(location_visits_data)
    location_visits_df.to_csv('location_visits.csv', index=False)
    print("Location visits saved to location_visits.csv")
    
    # Look for unusual patterns
    print("\nLooking for unusual patterns...")
    
    # Example: Employees who visited locations outside work hours
    unusual_patterns_data = []
    
    for emp_id, profile in profiles.items():
        late_night_visits = []
        for movement in profile['movement_patterns']:
            if movement['timestamp'].hour >= 22 or movement['timestamp'].hour <= 5:
                late_night_visits.append(movement)
                
                # Add to unusual patterns data
                unusual_patterns_data.append({
                    'employee': emp_id,
                    'pattern_type': 'Late night visit',
                    'timestamp': movement['timestamp'],
                    'location': movement['nearest_location'],
                    'latitude': movement['latitude'],
                    'longitude': movement['longitude']
                })
        
        if late_night_visits:
            print(f"{emp_id} has {len(late_night_visits)} late-night location visits")
    
    # Add other unusual patterns
    # For example, multiple visits to unusual locations
    location_frequency = {}
    for loc, count in location_counts.items():
        location_frequency[loc] = count / len(profiles)
    
    for emp_id, profile in profiles.items():
        emp_locations = {}
        for location in profile['locations_visited']:
            # Count this employee's visits to this location
            visit_count = sum(1 for m in profile['movement_patterns'] 
                            if m['nearest_location'] == location)
            emp_locations[location] = visit_count
        
        # Check for locations this employee visits much more than others
        for loc, count in emp_locations.items():
            if count > 3 * location_frequency.get(loc, 0) and location_frequency.get(loc, 0) > 0:
                # This employee visits this location 3x more than average
                unusual_patterns_data.append({
                    'employee': emp_id,
                    'pattern_type': 'Unusually frequent visits',
                    'timestamp': None,
                    'location': loc,
                    'latitude': None,
                    'longitude': None,
                    'visit_count': count,
                    'average_visits': location_frequency.get(loc, 0)
                })
                print(f"{emp_id} visits {loc} unusually frequently ({count} times vs avg {location_frequency.get(loc, 0):.2f})")
    
    # Convert to DataFrame and save to CSV
    unusual_patterns_df = pd.DataFrame(unusual_patterns_data)
    unusual_patterns_df.to_csv('unusual_patterns.csv', index=False)
    print("Unusual patterns saved to unusual_patterns.csv")
    
    # More analyses can be added here based on project requirements
    return location_visits_df, unusual_patterns_df

# Run the analysis and get the dataframes
location_visits_df, unusual_patterns_df = analyze_results(employee_profiles)

def visualize_clusters(coords, labels, clustering_model=None, known_locations=None):
    # Create a scatterplot of the clusters
    plt.figure(figsize=(12, 10))

    # Make sure both are numpy arrays with compatible types
    coords = np.array(coords)
    labels = np.array(labels, dtype=int)  # Ensure labels are integers
    
    # Debug information
    print(f"Coords shape: {coords.shape}, Labels shape: {labels.shape}")
    print(f"Label range: min={labels.min()}, max={labels.max()}")
    
    # Convert to 2D array if it's 1D
    if coords.ndim == 1 and len(coords) > 0:
        print("Detected 1D coordinates array, reshaping...")
        n_points = coords.shape[0] // 2
        coords = coords.reshape(n_points, 2)
    

    # Plot all points with larger size and more opacity
    scatter = plt.scatter(
        coords[:, 1],  # longitude (x-axis)
        coords[:, 0],  # latitude (y-axis)
        c=labels,
        cmap='viridis',  # Use a perceptually uniform colormap
        s=20,  # Increased size
        alpha=0.9  # Increased opacity
    )

    # Plot the cluster centers
    if clustering_model is not None:
        try:
            centers = clustering_model.cluster_centers_
            
            # Create a mapping from cluster index to location name (if any)
            cluster_to_location = {}
            if known_locations:
                for loc_name, (lat, lon) in known_locations.items():
                    # Only map if loc_name is not a generic name
                    if not loc_name.startswith("Location "):
                        min_dist = float('inf')
                        closest_cluster = None
                        for i, center in enumerate(centers):
                            dist = ((center[0] - lat)**2 + (center[1] - lon)**2)**0.5
                            if dist < min_dist:
                                min_dist = dist
                                closest_cluster = i
                        if min_dist < 0.02:  # Threshold for matching
                            cluster_to_location[closest_cluster] = loc_name
            
            # Plot unmapped clusters with 'x'
            unmapped_centers = [centers[i] for i in range(len(centers)) if i not in cluster_to_location]
            if unmapped_centers:
                unmapped_centers = np.array(unmapped_centers)
                plt.scatter(unmapped_centers[:, 1], unmapped_centers[:, 0], 
                           c='black', s=50, marker='x', label='Unmapped locations')
            
            # Plot mapped clusters with 'o'
            mapped_centers = [centers[i] for i in cluster_to_location.keys()]
            if mapped_centers:
                mapped_centers = np.array(mapped_centers)
                plt.scatter(mapped_centers[:, 1], mapped_centers[:, 0], 
                           c='gray', s=50, marker='o', facecolors='none', 
                           linewidth=1, label='Named locations')
                
                # Add labels for mapped locations
                for i, loc_name in cluster_to_location.items():
                    plt.annotate(loc_name, (centers[i][1], centers[i][0]),
                                xytext=(0, 5), textcoords='offset points')
                
        except (AttributeError, IndexError) as e:
            print(f"Could not plot cluster centers: {e}")
    
    # Add a legend
    plt.legend()
    
    # Add a colorbar to show the cluster mapping
    plt.colorbar(scatter, label='Cluster ID')
    
    plt.title('MiniBatchKMeans Clustering Results')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig('clustering_results.png', dpi=300)
    print("Clustering visualization saved as 'clustering_results.png'")
    plt.close()

def create_density_heatmap(coords):
    plt.figure(figsize=(12, 10))
    
    # Create a 2D histogram
    heatmap, xedges, yedges = np.histogram2d(
        coords[:, 1],  # longitude (x)
        coords[:, 0],  # latitude (y)
        bins=100
    )
    
    # Plot using imshow
    plt.imshow(
        heatmap.T,
        origin='lower',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap='hot',
        aspect='auto'
    )
    
    plt.colorbar(label='Point Density')
    plt.title('GPS Point Density')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig('gps_density.png', dpi=300)
    plt.close()

def visualize_clusters_interactive(coords, labels, clustering_model=None, known_locations=None):
    import numpy as np

    coords = np.array(coords)
    labels = np.array(labels, dtype=int)

    # Ensure coords is 2D and labels matches its length
    if coords.ndim == 1 and len(coords) > 0:
        n_points = coords.shape[0] // 2
        coords = coords.reshape(n_points, 2)
    if coords.shape[0] != labels.shape[0]:
        print(f"Shape mismatch: coords {coords.shape}, labels {labels.shape}")
        return

    # Prepare the scatter plot for all points
    fig = go.Figure()

    # Add all clustered points
    fig.add_trace(go.Scattergl(
        x=coords[:, 1],  # longitude
        y=coords[:, 0],  # latitude
        mode='markers',
        marker=dict(
            color=labels,
            colorscale='Viridis',
            size=10,
            opacity=1.0,
            showscale=True,
            colorbar=dict(title='Cluster ID')
        ),
        name='GPS Points',
        hoverinfo='skip'
    ))

    # Add cluster centers with hover labels if available
    if clustering_model is not None:
        centers = clustering_model.cluster_centers_
        hover_texts = []
        marker_colors = []
        # Build a reverse mapping: center index -> location name (from known_locations)
        center_to_location = {}
        if known_locations:
            for loc_name, (lat, lon) in known_locations.items():
                # Find the closest center to this location
                min_dist = float('inf')
                closest_idx = None
                for i, center in enumerate(centers):
                    dist = ((center[0] - lat) ** 2 + (center[1] - lon) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = i
                # Only assign if within threshold and not already assigned
                if min_dist < 0.02 and closest_idx is not None and closest_idx not in center_to_location:
                    center_to_location[closest_idx] = loc_name

        # Prepare hover texts and colors for each center
        for i, center in enumerate(centers):
            label = center_to_location.get(i, f"Cluster {i}")
            hover_texts.append(label)
            # Color red only if label is a real mapped location (not generic)
            if not (label.startswith("Cluster ") or label.startswith("Location ")):
                marker_colors.append('red')
            else:
                marker_colors.append('black')

        fig.add_trace(go.Scattergl(
            x=centers[:, 1],
            y=centers[:, 0],
            mode='markers',
            marker=dict(
                color=marker_colors,
                size=15,
                symbol='x'
            ),
            name='Cluster Centers',
            hovertext=hover_texts,
            hoverinfo='text'
        ))

    fig.update_layout(
        title='MiniBatchKMeans Clustering Results (Interactive)',
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        legend=dict(itemsizing='constant'),
        width=1000,
        height=800
    )

    fig.write_html('clustering_results_interactive.html')
    print("Interactive clustering visualization saved as 'clustering_results_interactive.html'")

# Usage:
visualize_clusters_interactive(clustering_coords, clustering_labels, clustering_model, known_locations)

create_density_heatmap(clustering_coords)


