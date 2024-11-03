import numpy as np
import math
from config import weapon_categories, min_x, max_x, min_y, max_y

def extract_single_element(lst):
    assert len(lst) == 1, "The input list must contain exactly one element."
    return lst[0]

def euclidean_distance(p1_coord, p2_coord):
    """Computes the euclidean distance between two points

    Args:
        p1_coord (float list): player 1 coordinate information
        p2_coord (float list): player 2 coordinate information

    Returns:
        distance: the euclidean distance between the two points
    """
    output = math.dist(p1_coord, p2_coord)
    # if output == 0:
    #     print(f"{p1_coord}, {p2_coord}")
    return output

def relative_distance_cross_to_enemy(pitch, yaw, X, Y, Z, enemy_X, enemy_Y, enemy_Z):
    """Computes the relative distance between the crosshair and the enemy player

    Args:
        pitch (np.float): pitch value of the player
        yaw (np.float): yaw value of the player
        X (np.float): X value of the player
        Y (np.float): Y value of the player
        Z (np.float): Z value of the player
        enemy_X (np.float): X value of the enemy player
        enemy_Y (np.float): Y value of the enemy player
        enemy_Z (np.float): Z value of the enemy player
        
    Returns:
        diff_x (np.float): difference in x direction
        diff_y (np.float): difference in y direction
        diff_z (np.float): difference in z direction
        distance (np.float): distance between the crosshair and the enemy player
    """
    crosshair_angle_x = np.cos(np.radians(yaw))
    crosshair_angle_y = np.sin(np.radians(yaw))
    crosshair_angle_z = np.sin(np.radians(pitch))
    
    # enemy_X - X is 
    dx = (enemy_X - X)
    dy = (enemy_Y - Y)
    dz = (enemy_Z - Z)
    
    real_angle_x = np.cos(np.radians(np.arctan2(dy, dx)))
    real_angle_y = np.sin(np.radians(np.arctan2(dy, dx)))
    real_angle_z = np.sin(np.radians(np.arctan2(dz, np.sqrt(dx**2 + dy**2))))
    
    diff_x = crosshair_angle_x - real_angle_x
    diff_y = crosshair_angle_y - real_angle_y
    diff_z = crosshair_angle_z - real_angle_z
    
    distance = np.sqrt((X-enemy_X)**2 + (Y-enemy_Y)**2 + (Z-enemy_Z)**2)

    return diff_x, diff_y, diff_z, distance

def pitch_yaw_diff_from_enemy(pitch, yaw, X, Y, Z, enemy_X, enemy_Y, enemy_Z):
    """Computes how far the crosshair is from the enemy
    
    Args:
        pitch (np.float or list): pitch value(s) of the player
        yaw (np.float or list): yaw value(s) of the player
        X (np.float or list): X value(s) of the player
        Y (np.float or list): Y value(s) of the player
        Z (np.float or list): Z value(s) of the player
        enemy_X (np.float or list): X value(s) of the enemy player
        enemy_Y (np.float or list): Y value(s) of the enemy player
        enemy_Z (np.float or list): Z value(s) of the enemy player

    Returns:
    
    """
    if isinstance(pitch, list) and isinstance(X, list):
        pitch_diff = []
        yaw_diff = []
        for i in range(len(pitch)):
            actual_p = np.degrees(np.arctan2(enemy_Z[i] - Z[i], np.sqrt((enemy_X[i] - X[i])**2 + (enemy_Y[i] - Y[i])**2)))
            actual_y = np.degrees(np.arctan2(enemy_Y[i] - Y[i], enemy_X[i] - X[i]))
        
            pitch_diff.append(actual_p - pitch[i])
            yaw_diff.append(actual_y - yaw[i])

    else:
        actual_pitch = np.degrees(np.arctan2(enemy_Z - Z, np.sqrt((enemy_X - X)**2 + (enemy_Y - Y)**2)))
        actual_yaw = np.degrees(np.arctan2(enemy_Y - Y, enemy_X - X))
                
        pitch_diff = actual_pitch - pitch
        yaw_diff = actual_yaw - yaw
        
    
    return pitch_diff, yaw_diff



def categorize_item(item):
    if not item:
        return '0'
    elif item and (item.startswith('knife') or item == 'bayonet'):
        return 'knife'
    else:
        return item
    
def weapon_one_hot_vector(item):
    category = categorize_item(item)
    one_hot_vector = [1 if cat == category else 0 for cat in weapon_categories]
    return one_hot_vector

def grab_X_info(start_index, end_index, df):
    # Grab only the start_index row
    X_info = []
    X_info.append(df.loc[start_index]['X'])
    X_info.append(df.loc[start_index]['Y'])
    X_info.append(df.loc[start_index]['Z'])
    X_info.append(df.loc[start_index]['health'])
    armor_value = df.loc[start_index]['armor_value']
    X_info.append(1 if armor_value > 0 else 0)
    X_info.append(df.loc[start_index]['has_helmet'])
    X_info.append(df.loc[start_index]['pitch'])
    X_info.append(df.loc[start_index]['yaw'])
    X_info.append(df.loc[start_index]['flash_duration'])
    weapon = weapon_one_hot_vector(df.loc[start_index]['active_weapon_name'])
    X_info += weapon
    return [X_info]

def add_pitch_diff(X_vec_0, X_vec_1):
    att_X, att_Y, att_Z = X_vec_0[:3]
    oppo_X, oppo_Y, oppo_Z = X_vec_1[:3]
    att_pitch_diff = pitch_yaw_diff_from_enemy(X_vec_0[6], X_vec_0[7], att_X, att_Y, att_Z, oppo_X, oppo_Y, oppo_Z)
    oppo_pitch_diff = pitch_yaw_diff_from_enemy(X_vec_1[6], X_vec_1[7], oppo_X, oppo_Y, oppo_Z, att_X, att_Y, att_Z)
    X_vec_0 += list(att_pitch_diff)
    X_vec_1 += list(oppo_pitch_diff)
    
    
def get_teams(df):
    team_0 = []
    team_1 = []
    the_range = int(len(df)/2)
    # subset the first 100 rows
    df = df[the_range:the_range+100]
    team_dict = df.groupby('team_num')['name'].apply(set).to_dict()
    print(team_dict)
    try:
        team_0 = list(team_dict[2])
        team_1 = list(team_dict[3])
    except KeyError:
        return None, None
    return team_0, team_1

def compute_entropy(freq_dict):
    total = sum(freq_dict.values())
    
    entropy = 0
    for count in freq_dict.values():
        if count > 0:
            prob = count / total
            entropy -= prob * math.log2(prob)
    
    return entropy

def generate_pixel_map_to_pixel_map(side_length=50):
    """
    Generates a pixel map of the game area, where each value in the dict is also a pixel map
    
    Args:
        side_length (int): the side length of the pixel map
        
    Returns:
        pixel_map (dict): a dictionary of the pixel map
    """
    x_coords = list(range(min_x, max_x+1, side_length))
    y_coords = list(range(min_y, max_y+1, side_length))
    square_ids = [(x, y) for x in x_coords for y in y_coords]
    square_dict = {square_id: {} for square_id in square_ids}
    return square_dict