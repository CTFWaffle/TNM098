import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import matplotlib.image as mpimg
import os

shared_locations = pd.read_csv(r'Projekt\data\MC2\shared_locations.csv', encoding='cp1252')

# Save the date from the 'time' column in a different column
shared_locations['date'] = pd.to_datetime(shared_locations['time']).dt.date
# Remove the date from the 'time' column
shared_locations['time'] = pd.to_datetime(shared_locations['time']).dt.time

# Create a datetime.time object for 18:00
time_threshold = datetime.time(18, 0)
filtered_instances = shared_locations[shared_locations['time'] >= time_threshold]

def plot_evening_locations(data, img_path=r'Projekt\data\MC2\MC2-tourist.jpg', 
                          title="Shared Locations After 18:00"):
    """
    Plot shared locations that occurred after 18:00 on the map
    
    Parameters:
    -----------
    data: DataFrame
        Filtered shared locations data after 18:00
    img_path: str
        Path to the map image
    title: str
        Title for the plot
    """
    # Load the map image
    img = mpimg.imread(img_path)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get approximate bounds from the data
    minx, maxx = data['x'].min(), data['x'].max()
    miny, maxy = data['y'].min(), data['y'].max()
    
    # Add some padding
    padding = 0.001  # Adjust as needed
    minx -= padding
    maxx += padding
    miny -= padding
    maxy += padding
    
    # Display the background image
    ax.imshow(img, extent=[minx, maxx, miny, maxy])
    
    # Create a colormap based on the number of people
    scatter = ax.scatter(
        data['x'], 
        data['y'], 
        c=data['num_people'], 
        cmap='viridis',
        alpha=0.7,
        s=data['num_people'] * 20,  # Scale point size by number of people
        edgecolor='black'
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Number of People')
    
    # Add annotations for points with more people
    threshold = 3  # Adjust as needed
    for _, row in data[data['num_people'] >= threshold].iterrows():
        # Extract just the IDs for display
        ids = row['people_ids'].split(', ')
        id_text = ', '.join(ids[:3]) + ('...' if len(ids) > 3 else '')
        
        ax.annotate(
            f"{row['time'].strftime('%H:%M')}\n{row['num_people']} people\n{id_text}",
            (row['x'], row['y']),
            textcoords="offset points", 
            xytext=(5, 5),
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.7),
            fontsize=8
        )
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(alpha=0.3)
    
    plt.show()
    
    # Return the figure object for further customization if needed
    return fig, ax

# Load data
shared_locations = pd.read_csv(r'Projekt\data\MC2\shared_locations.csv', encoding='cp1252')
    
# Process time columns
shared_locations['date'] = pd.to_datetime(shared_locations['time']).dt.date
shared_locations['time'] = pd.to_datetime(shared_locations['time']).dt.time
    
# Filter for times after 18:00
time_threshold = datetime.time(18, 0)
filtered_instances = shared_locations[shared_locations['time'] >= time_threshold]
    
print(f"Found {len(filtered_instances)} shared locations after 18:00")
    
# Plot the filtered data
plot_evening_locations(filtered_instances)

# Split the people names into separate columns
people_names = filtered_instances['people_ids'].str.split(', ', expand=True)
# Rename the columns
people_names.columns = [f'Person_{i+1}' for i in range(people_names.shape[1])]
# Concatenate the new columns with the original DataFrame
filtered_instances = pd.concat([filtered_instances, people_names], axis=1)

print(filtered_instances['people_ids'].unique())

print(len(filtered_instances['people_ids'].unique()))

print(filtered_instances[filtered_instances['x']>=24.89])

print(filtered_instances[filtered_instances['x']<=24.868])