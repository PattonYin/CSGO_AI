import numpy as np
# import torch



import sys
import os

# Add the parent directory of demo_analysis to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preperator import *
from config import *


params = dict(
    demo_path = "demo_analysis/demo/match730_003673760416913162325_1520768583_129.dem",
    player_name = "BERNARDO",
    columns = ["pitch", "yaw", "X", "Y", "Z", "shots_fired", "team_num"]
)

def visualize_single_player_trajs(demo_data, round_nums, player_id, clip_ticks=None):
    plt.figure(figsize=(10, 8))
    
    for round_num in round_nums:
        data = demo_data[(demo_data['round_num'] == round_num) & (demo_data['name'] == player_id)]
        print(f"Round {round_num}: Number of entries after filtering by player ID: {len(data)}")
        
        if clip_ticks is not None:
            data = data.iloc[clip_ticks[0]:clip_ticks[1]]
        
        X = data['X'].to_numpy()
        Y = data['Y'].to_numpy()
        
        plt.plot(X, Y, marker='o', linestyle='-', markersize=5, label=f'Round {round_num}')
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Player {player_id} Trajectories in Rounds {round_nums}')
    
    plt.xlim([min_x, max_x])
    plt.ylim([min_y, max_y])
    plt.grid(True)
    plt.legend()
    
    plt.show()
    
    
    pass
if __name__ == "__main__":
    processor = Processor(params["demo_path"])
    data = processor.get_round_data(props=params['columns'])
    print(data.name.unique())
    visualize_single_player_trajs(data, [1,2,3], "Bonks", clip_ticks=[0, 2000])