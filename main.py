import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Read Eyetracking data
eyetracking_data = pd.read_table('EyeTrack-raw.tsv')
print(eyetracking_data.head())

# Konvertera timestamp till sekunder
eyetracking_data['Time (s)'] = eyetracking_data['RecordingTimestamp'] / 1e6

# # Plotting GazePointX
# plt.figure(figsize=(10, 5))
# plt.plot(eyetracking_data['Time (s)'], eyetracking_data['GazePointX(px)'], label='Gaze X', alpha=0.7)
# plt.xlabel('Time (s)')
# plt.ylabel('Gaze X Position (px)')
# plt.title('Gaze X Position Over Time')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Scatter plot av X och Y koordinater
plt.figure(figsize=(8, 6))
plt.scatter(
    eyetracking_data['GazePointX(px)'],
    eyetracking_data['GazePointY(px)'],
    s=5,  # punktstorlek
    alpha=0.5  # transparens
)
plt.xlabel('Gaze X Position (px)')
plt.ylabel('Gaze Y Position (px)')
plt.title('Gaze Positions (Scatter Plot)')
plt.gca().invert_yaxis()  # Invertera Y för att matcha skärmkoordinater
plt.grid(True)
plt.tight_layout()
plt.show()

# Skapa heatmap med 2D histogram
plt.figure(figsize=(8, 6))
plt.hist2d(
    eyetracking_data['GazePointX(px)'],
    eyetracking_data['GazePointY(px)'],
    bins=100,  # finare upplösning
    cmap='hot'  # färgskala: 'hot', 'viridis', 'plasma' etc.
)
plt.xlabel('Gaze X Position (px)')
plt.ylabel('Gaze Y Position (px)')
plt.title('Gaze Heatmap')
plt.gca().invert_yaxis()  # matcha skärmkoordinater
plt.colorbar(label='Number of Gaze Points')
plt.tight_layout()
plt.show()

# # Förbered data
# x = eyetracking_data['GazePointX(px)'].dropna()
# y = eyetracking_data['GazePointY(px)'].dropna()
# t = eyetracking_data['RecordingTimestamp'].dropna() / 1e6  # sekunder

# # Trimma till lika längd
# min_len = min(len(x), len(y), len(t))
# x, y, t = x[:min_len], y[:min_len], t[:min_len]

# # 3D-plot (Space-Time Cube)
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, t, c=t, cmap='viridis', s=3, alpha=0.6)

# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Time (s)')
# ax.invert_yaxis()
# ax.set_title("Space-Time Cube: Gaze Over Time")
# plt.tight_layout()
# plt.show()

# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# # Data för animation
# x = eyetracking_data['GazePointX(px)'].dropna().values
# y = eyetracking_data['GazePointY(px)'].dropna().values

# fig, ax = plt.subplots(figsize=(8, 6))
# scat = ax.scatter([], [], s=10, c='red')
# ax.invert_yaxis()
# ax.set_xlim(min(x), max(x))
# ax.set_ylim(min(y), max(y))
# ax.set_title('Eye Gaze Over Time (Animated)')

# def update(frame):
#     scat.set_offsets(np.column_stack((x[:frame], y[:frame])))
#     return scat,

# ani = animation.FuncAnimation(fig, update, frames=len(x), interval=10, blit=True)
# plt.show()

# Ta ut ögonrörelsedata
x = eyetracking_data['GazePointX(px)'].dropna().values
y = eyetracking_data['GazePointY(px)'].dropna().values
t = eyetracking_data['RecordingTimestamp'].dropna().values / 1e6  # sekunder

# Säkerställ samma längd
n = min(len(x), len(y), len(t))
x, y, t = x[:n], y[:n], t[:n]

# Skapa figure och axlar
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(np.min(y), np.max(y))
ax.set_zlim(np.min(t), np.max(t))
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Time (s)")
ax.invert_yaxis()
ax.set_title("Animated Space-Time Cube with Transitions")

# Init tom linje
line, = ax.plot([], [], [], lw=2, color='blue')
point, = ax.plot([], [], [], 'ro')

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    point.set_data([], [])
    point.set_3d_properties([])
    return line, point

def update(frame):
    line.set_data(x[:frame], y[:frame])
    line.set_3d_properties(t[:frame])
    point.set_data(x[frame-1:frame], y[frame-1:frame])
    point.set_3d_properties(t[frame-1:frame])
    return line, point

ani = animation.FuncAnimation(fig, update, frames=n, init_func=init, interval=15, blit=True)
plt.show()