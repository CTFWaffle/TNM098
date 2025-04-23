import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

# Read Eyetracking data
eyetracking_data = pd.read_table('EyeTrack-raw.tsv')
print(eyetracking_data.head())

# Create a KDE-based heatmap
x = eyetracking_data['GazePointX(px)'].values
y = eyetracking_data['GazePointY(px)'].values
k = gaussian_kde([x, y])

# Create a grid to evaluate the KDE on
xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))

# Plot the heatmap
plt.figure(figsize=(10, 6))
plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='hot')
plt.colorbar(label='Density')
plt.title('Eyetracking Heatmap')
plt.xlabel('X position (pixels)')
plt.ylabel('Y position (pixels)')
plt.show()

# Create animation of gaze movement
def create_gaze_animation():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(eyetracking_data['GazePointX(px)'].min(), eyetracking_data['GazePointX(px)'].max())
    ax.set_ylim(eyetracking_data['GazePointY(px)'].min(), eyetracking_data['GazePointY(px)'].max())
    ax.set_xlabel('X position (pixels)')
    ax.set_ylabel('Y position (pixels)')
    ax.set_title('Gaze Point Movement Over Time')
    
    # Use LineCollection without setting color initially
    line = LineCollection([], linewidths=6, alpha=0.7)
    ax.add_collection(line)
    point, = ax.plot([], [], 'ro', ms=6)
    
    # Choose a colormap - 'viridis', 'plasma', 'inferno', 'magma', 'jet', etc.
    cmap = plt.cm.plasma
    
    # Create legend handles
    from matplotlib.lines import Line2D
    
    # Create custom legend elements
    legend_elements = [
        Line2D([0], [0], color=cmap(0.01), lw=6, alpha=0.7, label='Oldest Gaze Path'),
        Line2D([0], [0], color=cmap(0.99), lw=6, alpha=0.7, label='Newest Gaze Path'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=8, label='Current Gaze Point')
    ]
    
    # Add the legend to the axis
    ax.legend(handles=legend_elements, loc='lower right')
    
    def init():
        line.set_segments([])
        point.set_data([], [])
        return line, point
    
    def animate(i):
        frame = min(i, len(eyetracking_data)-1)
        x = eyetracking_data['GazePointX(px)'].values[:frame+1]
        y = eyetracking_data['GazePointY(px)'].values[:frame+1]
        
        # Create segments from consecutive points
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create color array based on position in sequence
        if len(segments) > 0:
            # Normalize segment indices to [0,1] for colormap
            norm = plt.Normalize(0, len(segments))
            colors = [cmap(norm(i)) for i in range(len(segments))]
            line.set_color(colors)
            
        line.set_segments(segments)
        point.set_data(x[-1:], y[-1:])
        return line, point
    
    ani = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(eyetracking_data),
        interval=25, blit=True, repeat=False
    )
    
    # To save the animation (uncomment):
    # ani.save('gaze_animation.gif', writer='pillow')
    
    plt.show()
    return ani

# Call the function to create and display the animation
# Note: Keep a reference to the animation object to prevent garbage collection
anim = create_gaze_animation()



