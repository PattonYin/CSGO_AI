import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from scipy.stats import zscore

from config import *

def plot_positions(data: pd.DataFrame, map_name=None):
    
    fig, ax = plt.subplots()
    
    if map_name is not None:
        map_img = mpimg.imread(f'demo_analysis/radar_maps/{map_name}.png')
        ax.imshow(map_img, extent=[min_x, max_x, min_y, max_y])  # Match this extent to your plot limits
    
    ax.scatter(data["X"], data["Y"])
    plt.show()

def plot_tick(data: pd.DataFrame, map_name=None, data_all=None):
    
    fig, ax = plt.subplots()
    
    if map_name is not None:
        map_img = mpimg.imread(f'demo_analysis/radar_maps/{map_name}.png')
        ax.imshow(map_img, extent=[min_x, max_x, min_y, max_y])  # Match this extent to your plot limits

    if data_all is not None:
        ax.scatter(data_all["X"], data_all["Y"], alpha=0.3)  # Players' positions

    scale = 50
    sizes = [20 if name == 'BERNARDO' else 10 for name in data['name']]
    colors = ['blue' if team_num == 3 else 'red' for team_num in data['team_num']]
    ax.scatter(data["X"], data["Y"], s=sizes, color=colors)  # Players' positions
    
    # Calculate the end points for the lines
    for i, row in data.iterrows():
        # Calculate the change in X and Y based on yaw and a fixed scale
        dx = scale * np.cos(np.radians(row['yaw']))
        dy = scale * np.sin(np.radians(row['yaw']))

        # Draw the line from the player's position (X, Y) to the calculated point (X + dx, Y + dy)
        ax.arrow(row['X'], row['Y'], dx, dy, head_width=0.1*scale, head_length=0.2*scale, fc='red', ec='red')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Player Facing Directions')
    ax.grid(True)
    ax.set_aspect('equal')  # Ensures that scale is consistent across x and y axis
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    plt.show()
    

def plot_move(data: pd.DataFrame, tick_rate=32, map_name=None, data_all=None):
    data.reset_index(drop=True, inplace=True)
    start_tick = data.loc[0, 'tick']
    
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = ax.plot([], [], 'ro')
    
    ax.scatter(data["X"], data["Y"])
    
    def init():
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        return ln,
    
    def update(frame):
        # xdata.append(frame)
        xdata = data[data['tick'] == start_tick + tick_rate*frame]['X'].tolist()
        # ydata.append(np.sin(frame))
        ydata = data[data['tick'] == start_tick + tick_rate*frame]['Y'].tolist()
        ln.set_data(xdata, ydata)
        return ln,
    
    ani = FuncAnimation(fig, update, frames=list(range(1,(data.tick.nunique()//tick_rate))),
                        init_func=init, blit=True)
    plt.show()
    
def plot_pixel_map(data: dict, side_length, title="Visualization"):

    step_x = step_y = side_length

    grid_x = (max_x - min_x) // step_x + 1
    grid_y = (max_y - min_y) // step_y + 1
    value_array = []

    grid = np.full((grid_y, grid_x), np.nan)
    for (x, y), value in data.items():
        idx_x = (x - min_x) // step_x
        idx_y = (y - min_y) // step_y
        if value is not None:
            value_array.append(value)
            grid[idx_y, idx_x] = value

    valid_mask = ~np.isnan(grid)
    grid[valid_mask] = zscore(grid[valid_mask])

    fig, ax = plt.subplots()
    cmap = plt.get_cmap('Blues')  
    cmap.set_bad(color='grey') 
    
    cax = ax.imshow(grid, interpolation='nearest', cmap=cmap, origin='lower', extent=(min_x, max_x, min_y, max_y))
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
    cbar.set_label('Standardized Value')
    plt.title(title)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.show()