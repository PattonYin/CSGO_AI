from data_preperator import Processor
from models import Expected_Win_Rate
from config import data_map_X_1
from plot_module import plot_player_with_direction

import numpy as np


wanted_props = ['team_num', 'flash_duration', 'pitch', 'yaw', 'X', 'Y', 'Z', 'armor_value', 'active_weapon_name', 'flash_max_alpha', 'health', 'is_alive', 'has_helmet', 'move_type']

def yaw_diff():
    # params
    data_map = data_map_X_1
    data_path = r"intermediate_data\model_training\10"
    Loader = Expected_Win_Rate()
    
    # Get the data
    X, Y = Loader.load_data(data_path=data_path)
    
    # Get the pitch and yaw data for attacker and attackee
    index_checkout = 0
    x0 = X.loc[index_checkout]
    mid_point = data_map['mid_point']

    player_1_pitch_yaw = [float(x0[i]) for i in data_map['pitch_&_yaw']]
    player_2_pitch_yaw = [float(x0[i+mid_point]) for i in data_map['pitch_&_yaw']]
    
    player_1_xyz = [float(x0[i]) for i in data_map['player_coordinates']]
    player_2_xyz = [float(x0[i+mid_point]) for i in data_map['player_coordinates']]
    
    # Plot the data
    pitch_yaws = [player_1_pitch_yaw, player_2_pitch_yaw]
    xyzs = [player_1_xyz, player_2_xyz]
    
    plot_player_with_direction(xyzs, pitch_yaws)
    
    # Compute the diff
    actual_yaw1 = np.degrees(np.arctan2(player_2_xyz[1] - player_1_xyz[1], player_2_xyz[0] - player_1_xyz[0]))
    
    actual_yaw2 = np.degrees(np.arctan2(player_1_xyz[1] - player_2_xyz[1], player_1_xyz[0] - player_2_xyz[0]))
    
    # Plot the diff
    plot_player_with_direction(xyzs, [[0, actual_yaw1], [0, actual_yaw2]])
    
    
    pass


if __name__ == "__main__":
    yaw_diff()
