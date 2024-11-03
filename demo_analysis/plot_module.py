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
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
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
        ax.text(row['X'], row['Y'], f"{row['pitch']:.2f}, {row['yaw']:.2f}", fontsize=8, color='black')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Player Facing Directions')
    ax.grid(True)
    ax.set_aspect('equal')  # Ensures that scale is consistent across x and y axis
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    plt.show()
    
def plot_player_with_direction(player_xyzs, player_pitch_yaws, color='blue'):
    plt.figure(figsize=(10, 10))
    
    for index, player_xyz in enumerate(player_xyzs):
        player_pitch_yaw = player_pitch_yaws[index]

        x, y, z = player_xyz
        pitch, yaw = player_pitch_yaw
        
        yaw_radians = np.radians(yaw)
        
        arrow_length = 1.0 
        dx = arrow_length * np.cos(yaw_radians)
        dy = arrow_length * np.sin(yaw_radians)
        
        plt.scatter(x, y, color=color, label=str(index))
        plt.arrow(x, y, dx, dy, head_width=0.2, head_length=0.2, fc=color, ec=color)
        
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    # plt.xlim(min_x, max_x)
    # plt.ylim(min_y, max_y)
    plt.legend()
    plt.grid(True)
    
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
    
def plot_move_seq(seq, data_map): 
    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = ax.plot([], [], 'ro')
    
    xs = seq[:, [data_map['player_coordinates'][0], data_map['player_coordinates'][0]+data_map['mid_point']]]
    ys = seq[:, [data_map['player_coordinates'][1], data_map['player_coordinates'][0]+data_map['mid_point']]]
        
    def init():
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        return ln,
    
    def update(frame):
        xdata = xs[:frame + 1]
        ydata = ys[:frame + 1]
        ln.set_data(xdata, ydata)
        return ln,
    
    ani = FuncAnimation(fig, update, frames=list(range(0,len(xs))),
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
    # cmap = plt.get_cmap('seismic')
    cmap.set_bad(color='grey') 
    
    cax = ax.imshow(grid, interpolation='nearest', cmap=cmap, origin='lower', extent=(min_x, max_x, min_y, max_y))
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
    cbar.set_label('Standardized Value')
    plt.title(title)
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.show()
    
def visualize_X(X, predicted_prob, title="Visualization"):
    data_map = data_map_X_1
    mid_point = data_map['mid_point']
    
    plt.figure(figsize=(6, 6))
    
    def visualize_yaw(coords, yaw, color="black"):
        x,y,z = coords
        arrow_length = 100
        
        dx = arrow_length * np.cos(np.radians(yaw))
        dy = arrow_length * np.sin(np.radians(yaw))
        
        plt.arrow(x, y, dx, dy, head_width=40, head_length=40, fc=color, ec=color)
        
    # Plot the attacker and attackee
    attacker_coord = X[data_map['player_coordinates'][0]:data_map['player_coordinates'][-1]+1]
    attacker_yaw = X[data_map['pitch_&_yaw'][1]]
    
    attackee_coord = X[data_map['player_coordinates'][0]+mid_point:data_map['player_coordinates'][-1]+mid_point+1]
    attackee_yaw = X[data_map['pitch_&_yaw'][1]+mid_point]

    plt.scatter(attacker_coord[0], attacker_coord[1], color='red', label='winner', s=40)
    plt.scatter(attackee_coord[0], attackee_coord[1], color='blue', label='loser', s=40)

    visualize_yaw(attacker_coord, attacker_yaw, 'red')
    visualize_yaw(attackee_coord, attackee_yaw, 'blue')
    
    # Plot teammates coordinates
    attacker_teammates_coords = X[data_map['teammate_coordinates'][0]:data_map['teammate_coordinates'][-1]]
    attackee_teammates_coords = X[data_map['teammate_coordinates'][0]+mid_point:data_map['teammate_coordinates'][-1]+mid_point]
    
    # Visualize coordinates
    for index in range(len(attacker_teammates_coords)//3):
        plt.scatter(attacker_teammates_coords[index*3], attacker_teammates_coords[index*3+1], color='red', label='winner teammates', s=10)
    
    for coords in range(len(attackee_teammates_coords)//3):
        plt.scatter(attackee_teammates_coords[index*3], attackee_teammates_coords[index*3+1], color='blue', label='loser teammates', s=10)
        
    # Draw text next to the player to indicate the additional information
    att_health = X[data_map['player_health'][0]]
    oppo_health = X[data_map['player_health'][0]+mid_point]
    att_armor = X[data_map['armor_value'][0]]
    oppo_armor = X[data_map['armor_value'][0]+mid_point]
    att_helmet = X[data_map['has_helmet'][0]]
    oppo_helmet = X[data_map['has_helmet'][0]+mid_point]
    att_flash = X[data_map['flash_duration'][0]]
    oppo_flash = X[data_map['flash_duration'][0]+mid_point]
    
    fontsize = 6
    plt.text(attacker_coord[0]-200, attacker_coord[1]+120, f'Health: {att_health}\nArmor: {att_armor}\nHelmet: {att_helmet}\nFlash: {att_flash:.2f}', fontsize=fontsize)
    plt.text(attackee_coord[0]-200, attackee_coord[1]+120, f'Health: {oppo_health}\nArmor: {oppo_armor}\nHelmet: {oppo_helmet}\nFlash: {oppo_flash:.2f}', fontsize=fontsize)
    
    plt.text(attacker_coord[0]-720, attacker_coord[1]-360, f'xWinRate: {predicted_prob:.3f}', fontsize=fontsize*1.5)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.title(title)
    # plt.legend()
    # plt.grid(True)
    
    plt.show()
    
    
def player_xWinRate_vs_actualWinRate(stats_dict):
    players = list(stats_dict.keys())
    
    # Extract xWinRate and actualWinRate for each player
    xWinRates = [stats[0] for stats in stats_dict.values()]
    actualWinRates = [stats[1] for stats in stats_dict.values()]
    
    num_players = len(players)
    
    ind = np.arange(num_players)  # the x locations for the groups
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(8, 6))

    rects1 = ax.bar(ind - width/2, xWinRates, width, label='xWinRate')

    rects2 = ax.bar(ind + width/2, actualWinRates, width, label='Actual WinRate')

    ax.set_xlabel('Players')
    ax.set_ylabel('Win Rates')
    ax.set_title('xWinRate vs Actual WinRate by Player')
    ax.set_xticks(ind)
    ax.set_xticklabels(players)
    ax.legend()

    plt.xticks(rotation=45)  # Rotate player names for better readability
    plt.tight_layout()
    plt.show()