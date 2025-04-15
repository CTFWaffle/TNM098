import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.cluster import KMeans

# Read Eyetracking data
eyetracking_data = pd.read_csv('EyeTrack-raw.tsv', sep='\t')
eyetracking_data['x'] = eyetracking_data['GazePointX(px)']
eyetracking_data['y'] = eyetracking_data['GazePointY(px)']

# Calculate Kmeans clustering for the x and y coordinates
kmeans = KMeans(n_clusters=4, random_state=0).fit(eyetracking_data[['x', 'y']])

#scatter plot of x and y
plt.scatter(eyetracking_data['x'], eyetracking_data['y'], c=kmeans.labels_, cmap='viridis', marker='o')
plt.colorbar(label='Cluster Label')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Scatter plot of Eye Tracking Data')
plt.show()

# Function to update the animation frame
def update(frame):
    plt.cla()  # Clear the current axes
    alpha_values = np.linspace(0.01, 1, frame + 1)  # Interpolated opacity values
    plt.plot(eyetracking_data['x'][:frame + 1], eyetracking_data['y'][:frame + 1], 
             marker='o', linestyle='-', markersize=2, alpha=0.5, color='blue')
    plt.scatter(eyetracking_data['x'][frame], eyetracking_data['y'][frame], 
                color='red', s=50, label='Current Point')  # Highlight current point
    plt.title('Eye Movement Path (Animated)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.gca().invert_yaxis()  # Invert y-axis to match screen coordinates
    plt.grid()
    plt.legend()

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 6))

# Create the animation
ani = FuncAnimation(fig, update, frames=len(eyetracking_data), interval=100, repeat=False)

# Save or display the animation
#ani.save('eye_movement_animation.mp4', writer='ffmpeg', fps=10)  # Save as video
plt.show()  # Display the animation


