import numpy as np

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
        
        # print(f"actual_pitch: {actual_pitch:.2f}, crosshair_pitch: {pitch:.2f}")
        
        pitch_diff = actual_pitch - pitch
        yaw_diff = actual_yaw - yaw
        
        # print(f"actual_yaw: {actual_yaw:.2f}, crosshair_yaw: {yaw:.2f}, yaw_diff: {yaw_diff:.2f}")

    
    return pitch_diff, yaw_diff