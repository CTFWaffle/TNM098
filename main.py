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
kmeans = KMeans(n_clusters=6, random_state=0).fit(eyetracking_data[['x', 'y']])

#scatter plot of x and y with Kmeans labels
plt.scatter(eyetracking_data['x'], eyetracking_data['y'], c=kmeans.labels_, cmap='viridis', marker='o')
plt.colorbar(label='Cluster Label')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Scatter plot of Eye Tracking Data')
plt.show()

