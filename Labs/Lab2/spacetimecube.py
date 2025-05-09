import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Read Eyetracking data
eyetracking_data = pd.read_table('Labs/Lab2/EyeTrack-raw.tsv')

# Konvertera timestamp till sekunder
eyetracking_data['Time (s)'] = eyetracking_data['RecordingTimestamp'] / 1e6

# Ta ut ögonrörelsedata
x = eyetracking_data['GazePointX(px)'].dropna().values
y = eyetracking_data['GazePointY(px)'].dropna().values
t = eyetracking_data['RecordingTimestamp'].dropna().values / 1e6  # sekunder

# Säkerställ samma längd
n = min(len(x), len(y), len(t))
x, y, t = x[:n], y[:n], t[:n]

# Plot the entire space-time cube with interpolated opacity
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Set axis limits and labels
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(np.min(y), np.max(y))
ax.set_zlim(np.min(t), np.max(t))
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Time (s)")
ax.invert_yaxis()
ax.set_title("Space-Time Cube with Interpolated Opacity")

# Plot the line with a fixed opacity
for i in range(1, len(x)):
    ax.plot(x[i-1:i+1], y[i-1:i+1], t[i-1:i+1], color='blue', alpha=0.2, lw=2)  # Set opacity to 0.5

# Plot all points
ax.scatter(x, y, t, color='red', s=10, label='Gaze Points', alpha=0.5)

plt.legend()
plt.show()