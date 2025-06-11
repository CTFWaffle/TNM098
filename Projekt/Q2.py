import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from sklearn.cluster import MiniBatchKMeans
import plotly.graph_objs as go
import plotly.express as px

# -------------------- Data Loading --------------------

# Define the data directory
data_dir = "Projekt/data/MC2/"

# Load credit card transaction data
cc_data = pd.read_csv(
    os.path.join(data_dir, "cc_data.csv"),
    parse_dates=['timestamp'],
    encoding='cp1252'
)

# Load loyalty card transaction data
loyalty_data = pd.read_csv(
    os.path.join(data_dir, "loyalty_data.csv"),
    parse_dates=['timestamp'],
    encoding='cp1252'
)

# Load GPS data
gps_data = pd.read_csv(
    os.path.join(data_dir, "gps.csv"),
    parse_dates=['Timestamp'],
    encoding='cp1252'
)

# Load car assignments (which employee drives which car)
car_assignments = pd.read_csv(
    os.path.join(data_dir, "car-assignments.csv"),
    encoding='cp1252'
)

# -------------------- Helper Functions --------------------

def preprocess_gps_data(gps_df):
    """Standardize GPS dataframe column names for easier processing."""
    gps_df = gps_df.rename(columns={
        'Timestamp': 'timestamp',
        'id': 'car_id',
        'lat': 'latitude',
        'long': 'longitude'
    })
    return gps_df

def get_nearest_location(lat, lon, location_dict):
    """
    Find the nearest known location to the given latitude and longitude.
    Returns the location name and the distance.
    """
    min_dist = float('inf')
    nearest_loc = None
    for loc_name, loc_coords in location_dict.items():
        loc_lat, loc_lon = loc_coords
        dist = ((lat - loc_lat)**2 + (lon - loc_lon)**2)**0.5
        if dist < min_dist:
            min_dist = dist
            nearest_loc = loc_name
    return nearest_loc, min_dist

def match_transaction_to_gps(transaction, gps_df, time_window=600):
    """
    Find GPS records within a time window (in seconds) of a transaction.
    Returns a DataFrame of nearby GPS records.
    """
    transaction_time = transaction['timestamp']
    time_lower = transaction_time - pd.Timedelta(seconds=time_window)
    time_upper = transaction_time + pd.Timedelta(seconds=time_window)
    nearby_gps = gps_df[
        (gps_df['timestamp'] >= time_lower) &
        (gps_df['timestamp'] <= time_upper)
    ]
    return nearby_gps

def link_transaction_to_employee(transaction, nearby_gps, car_assignments_df):
    """
    Link a transaction to an employee by matching car IDs in GPS data
    to car assignments. Returns (last_name, first_name) or (None, None).
    """
    if nearby_gps.empty:
        return None, None
    car_ids = nearby_gps['car_id'].unique()
    for car_id in car_ids:
        employee = car_assignments_df[car_assignments_df['CarID'] == car_id]
        if not employee.empty:
            return employee['LastName'].iloc[0], employee['FirstName'].iloc[0]
    return None, None

# -------------------- Clustering and Location Mapping --------------------

def identify_key_locations():
    """
    Use MiniBatchKMeans clustering to identify key locations from GPS data.
    Returns:
        - final_locations: dict of location name -> (lat, lon)
        - filtered_coords: coordinates used for clustering
        - labels: cluster labels for each point
        - clustering: fitted MiniBatchKMeans model
    """
    print("Identifying key locations with MiniBatchKMeans clustering...")

    # Extract latitude and longitude as numpy array
    coords = gps_data[['lat', 'long']].values
    print(f"Extracted {len(coords)} GPS points for clustering")

    # Downsample if dataset is too large
    max_points = 700000
    if len(coords) > max_points:
        print(f"Dataset too large, sampling {max_points} points from {len(coords)} total points")
        sample_idx = np.random.choice(len(coords), max_points, replace=False)
        coords = coords[sample_idx]
        print(f"Sampled down to {len(coords)} points")

    # Optional: filter outliers (currently commented out)
    # def filter_gps_outliers(coords):
    #     lat_q1, lat_q3 = np.percentile(coords[:, 0], [1, 99])
    #     lon_q1, lon_q3 = np.percentile(coords[:, 1], [1, 99])
    #     mask = (
    #         (coords[:, 0] >= lat_q1) & (coords[:, 0] <= lat_q3) &
    #         (coords[:, 1] >= lon_q1) & (coords[:, 1] <= lon_q3)
    #     )
    #     return coords[mask]
    # coords = filter_gps_outliers(coords)
    # print(f"After filtering outliers: {len(coords)} points")

    filtered_coords = coords  # Save for plotting

    # Cluster the coordinates
    n_clusters = 34
    clustering = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=10000,
        random_state=42
    ).fit(filtered_coords)

    labels = clustering.labels_
    n_clusters = len(set(labels))
    print(f"Found {n_clusters} distinct location clusters")

    # Group points by cluster
    clusters = {}
    for i, label in enumerate(labels):
        # MiniBatchKMeans does not use -1 for noise, but keep for compatibility
        if label == -1:
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(coords[i])

    # Calculate cluster centers as average of points in each cluster
    locations = {}
    for i, points in clusters.items():
        avg_lat = sum(p[0] for p in points) / len(points)
        avg_long = sum(p[1] for p in points) / len(points)
        locations[f"Location {i+1}"] = (avg_lat, avg_long)

    # Try to map clusters to real location names from transaction data
    location_mapping = map_clusters_to_names(locations, cc_data, loyalty_data)

    # Replace generic names with actual location names where possible
    final_locations = {}
    for loc_id, coords in locations.items():
        if loc_id in location_mapping:
            final_locations[location_mapping[loc_id]] = coords
        else:
            final_locations[loc_id] = coords

    return final_locations, filtered_coords, labels, clustering

def map_clusters_to_names(cluster_locations, cc_data, loyalty_data):
    """
    Greedily map each cluster to at most one real location name (from transactions).
    Returns a dict: cluster_id -> location_name
    """
    processed_gps = preprocess_gps_data(gps_data)
    location_mapping = {}

    # Get all unique location names from both transaction datasets
    all_locations = set(cc_data['location'].unique()) | set(loyalty_data['location'].unique())

    # For each named location, find average GPS position near its transactions
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

    # Sort all pairs by distance and greedily assign closest pairs (one-to-one)
    pairs.sort()
    used_locations = set()
    used_clusters = set()
    for dist, loc_name, cluster_id in pairs:
        if dist < 0.05 and loc_name not in used_locations and cluster_id not in used_clusters:
            location_mapping[cluster_id] = loc_name
            used_locations.add(loc_name)
            used_clusters.add(cluster_id)

    print(f"Successfully mapped {len(location_mapping)} clusters to named locations (one-to-one)")
    return location_mapping

# -------------------- Transaction & Employee Analysis --------------------

def analyze_transactions(known_locations):
    """
    Main analysis function to process all transactions and build employee profiles.
    Returns a dictionary of employee profiles.
    """
    processed_gps = preprocess_gps_data(gps_data)
    employee_profiles = {}

    # --- Process credit card transactions ---
    print("Processing credit card transactions...")
    for idx, transaction in cc_data.iterrows():
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
                    'movement_patterns': [],
                    'loyaltynum': None,
                    'last4ccnum': None
                }
            # Store the credit card number if not already set
            if employee_profiles[employee_id]['last4ccnum'] is None and 'last4ccnum' in transaction:
                employee_profiles[employee_id]['last4ccnum'] = transaction['last4ccnum']
            # Add transaction to profile
            employee_profiles[employee_id]['credit_transactions'].append({
                'timestamp': transaction['timestamp'],
                'location': transaction['location'],
                'price': transaction['price'],
                'car_id': nearby_gps['car_id'].iloc[0] if not nearby_gps.empty else None,
                'last4ccnum': transaction['last4ccnum'] if 'last4ccnum' in transaction else None
            })
            employee_profiles[employee_id]['total_spent'] += transaction['price']
            employee_profiles[employee_id]['locations_visited'].add(transaction['location'])

    # --- Process loyalty card transactions ---
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
                    'movement_patterns': [],
                    'loyaltynum': None,
                    'last4ccnum': None
                }
            # Store the loyalty number if not already set
            if employee_profiles[employee_id]['loyaltynum'] is None and 'loyaltynum' in transaction:
                employee_profiles[employee_id]['loyaltynum'] = transaction['loyaltynum']
            employee_profiles[employee_id]['loyalty_transactions'].append({
                'timestamp': transaction['timestamp'],
                'location': transaction['location'],
                'loyaltynum': transaction['loyaltynum'] if 'loyaltynum' in transaction else None,
                'price': transaction['price'] if 'price' in transaction else 0
            })
            employee_profiles[employee_id]['locations_visited'].add(transaction['location'])
            if 'price' in transaction:
                employee_profiles[employee_id]['total_spent'] += transaction['price']

    # --- Add GPS movement patterns to profiles ---
    print("Adding movement patterns...")
    for _, car_info in car_assignments.iterrows():
        employee_id = f"{car_info['FirstName']} {car_info['LastName']}"
        car_id = car_info['CarID']
        if employee_id not in employee_profiles:
            employee_profiles[employee_id] = {
                'credit_transactions': [],
                'loyalty_transactions': [],
                'locations_visited': set(),
                'total_spent': 0,
                'movement_patterns': [],
                'loyaltynum': None,
                'last4ccnum': None
            }
        # Get all GPS records for this car, sorted by time
        car_gps = processed_gps[processed_gps['car_id'] == car_id].sort_values('timestamp')
        for _, gps_record in car_gps.iterrows():
            nearest_loc, dist = get_nearest_location(
                gps_record['latitude'], gps_record['longitude'], known_locations
            )
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

# -------------------- Analysis --------------------

def analyze_results(profiles):
    """
    Analyze the results to identify patterns and anomalies.
    Saves location visits to CSV and prints popular locations.
    Returns a DataFrame of location visits.
    """
    # Count visits to each location
    location_counts = {}
    for emp_id, profile in profiles.items():
        for location in profile['locations_visited']:
            if location not in location_counts:
                location_counts[location] = 0
            location_counts[location] += 1

    # Print most popular locations
    popular_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)
    print("Most popular locations:")
    for loc, count in popular_locations[:10]:
        print(f"  {loc}: {count} visits")

    # Build a DataFrame of all location visits by employee
    location_visits_data = []
    for emp_id, profile in profiles.items():
        for location in profile['locations_visited']:
            visit_count = sum(
                1 for movement in profile['movement_patterns']
                if movement['nearest_location'] == location
            )
            location_visits_data.append({
                'employee': emp_id,
                'location': location,
                'visit_count': visit_count,
                'total_spent': (
                    sum(t['price'] for t in profile['credit_transactions'] if t['location'] == location) +
                    sum(t['price'] for t in profile['loyalty_transactions'] if t['location'] == location)
                )
            })
    location_visits_df = pd.DataFrame(location_visits_data)
    location_visits_df.to_csv('location_visits.csv', index=False)
    print("Location visits saved to location_visits.csv")
    return location_visits_df

# -------------------- Visualization --------------------

def visualize_clusters(coords, labels, clustering_model=None, known_locations=None):
    """
    Create a static scatterplot of the clusters and their centers.
    """
    plt.figure(figsize=(12, 10))
    coords = np.array(coords)
    labels = np.array(labels, dtype=int)
    print(f"Coords shape: {coords.shape}, Labels shape: {labels.shape}")
    print(f"Label range: min={labels.min()}, max={labels.max()}")

    # Reshape if needed
    if coords.ndim == 1 and len(coords) > 0:
        n_points = coords.shape[0] // 2
        coords = coords.reshape(n_points, 2)

    scatter = plt.scatter(
        coords[:, 1],  # longitude
        coords[:, 0],  # latitude
        c=labels,
        cmap='viridis',
        s=20,
        alpha=0.9
    )

    # Plot cluster centers and annotate with names if available
    if clustering_model is not None:
        try:
            centers = clustering_model.cluster_centers_
            cluster_to_location = {}
            if known_locations:
                for loc_name, (lat, lon) in known_locations.items():
                    if not loc_name.startswith("Location "):
                        min_dist = float('inf')
                        closest_cluster = None
                        for i, center in enumerate(centers):
                            dist = ((center[0] - lat)**2 + (center[1] - lon)**2)**0.5
                            if dist < min_dist:
                                min_dist = dist
                                closest_cluster = i
                        if min_dist < 0.02:
                            cluster_to_location[closest_cluster] = loc_name
            unmapped_centers = [centers[i] for i in range(len(centers)) if i not in cluster_to_location]
            if unmapped_centers:
                unmapped_centers = np.array(unmapped_centers)
                plt.scatter(unmapped_centers[:, 1], unmapped_centers[:, 0],
                            c='black', s=50, marker='x', label='Unmapped locations')
            mapped_centers = [centers[i] for i in cluster_to_location.keys()]
            if mapped_centers:
                mapped_centers = np.array(mapped_centers)
                plt.scatter(mapped_centers[:, 1], mapped_centers[:, 0],
                            c='gray', s=50, marker='o', facecolors='none',
                            linewidth=1, label='Named locations')
                for i, loc_name in cluster_to_location.items():
                    plt.annotate(loc_name, (centers[i][1], centers[i][0]),
                                 xytext=(0, 5), textcoords='offset points')
        except (AttributeError, IndexError) as e:
            print(f"Could not plot cluster centers: {e}")

    plt.legend()
    plt.colorbar(scatter, label='Cluster ID')
    plt.title('MiniBatchKMeans Clustering Results')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig('clustering_results.png', dpi=300)
    print("Clustering visualization saved as 'clustering_results.png'")
    plt.close()

def create_density_heatmap(coords):
    """
    Create and save a density heatmap of GPS points.
    """
    plt.figure(figsize=(12, 10))
    heatmap, xedges, yedges = np.histogram2d(
        coords[:, 1],  # longitude
        coords[:, 0],  # latitude
        bins=100
    )
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
    """
    Create an interactive Plotly visualization of clusters and centers.
    """
    coords = np.array(coords)
    labels = np.array(labels, dtype=int)
    if coords.ndim == 1 and len(coords) > 0:
        n_points = coords.shape[0] // 2
        coords = coords.reshape(n_points, 2)
    if coords.shape[0] != labels.shape[0]:
        print(f"Shape mismatch: coords {coords.shape}, labels {labels.shape}")
        return

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=coords[:, 1],
        y=coords[:, 0],
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
        center_to_location = {}
        if known_locations:
            for loc_name, (lat, lon) in known_locations.items():
                min_dist = float('inf')
                closest_idx = None
                for i, center in enumerate(centers):
                    dist = ((center[0] - lat) ** 2 + (center[1] - lon) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = i
                if min_dist < 0.02 and closest_idx is not None and closest_idx not in center_to_location:
                    center_to_location[closest_idx] = loc_name
        for i, center in enumerate(centers):
            label = center_to_location.get(i, f"Cluster {i}")
            hover_texts.append(label)
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
        legend=dict(
            itemsizing='constant',
            x=0.01,
            y=0.99,
            xanchor='left',
            yanchor='top'
        ),
        width=1000,
        height=800
    )
    fig.write_html('clustering_results_interactive.html')
    print("Interactive clustering visualization saved as 'clustering_results_interactive.html'")

# -------------------- DataFrame Display --------------------

def display_as_dataframe(profiles):
    """
    Convert employee profiles to a summary DataFrame and save to CSV.
    """
    summary_data = []
    for emp_id, profile in profiles.items():
        credit_spent = sum(transaction.get('price', 0)
                           for transaction in profile['credit_transactions'])
        loyalty_spent = sum(transaction.get('price', 0)
                            for transaction in profile['loyalty_transactions'])
        total_spent = credit_spent + loyalty_spent
        row = {
            'Employee': emp_id,
            'Credit Card Spent': credit_spent,
            'Loyalty Card Spent': loyalty_spent,
            'Total Spent': total_spent,
            'Credit Transactions': len(profile['credit_transactions']),
            'Loyalty Transactions': len(profile['loyalty_transactions']),
            'Locations Visited': len(profile['locations_visited']),
            'Movement Records': len(profile['movement_patterns']),
            'Loyalty Number': profile['loyaltynum'] if profile['loyaltynum'] is not None else 'N/A',
            'Credit Card Last 4': profile['last4ccnum'] if profile['last4ccnum'] is not None else 'N/A'
        }
        summary_data.append(row)
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('employee_summary.csv', index=False)
    print("\nEMPLOYEE SUMMARY")
    print(summary_df)

# -------------------- Main Execution --------------------

# Identify key locations using clustering
known_locations, clustering_coords, clustering_labels, clustering_model = identify_key_locations()

# Analyze all transactions and build employee profiles
employee_profiles = analyze_transactions(known_locations)

# Analyze results and save location visits
location_visits_df = analyze_results(employee_profiles)

# Visualizations
visualize_clusters_interactive(clustering_coords, clustering_labels, clustering_model, known_locations)
create_density_heatmap(clustering_coords)

# Display employee summary as DataFrame
display_as_dataframe(employee_profiles)


