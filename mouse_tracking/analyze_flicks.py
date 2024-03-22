import matplotlib.pyplot as plt

def read_data(file_path):
    x_coords = []
    y_coords = []
    timestamps = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 4 and parts[0] == 'mouse':
                x_coords.append(float(parts[1]))
                y_coords.append(float(parts[2]))
                timestamps.append(float(parts[3]))
    return x_coords, y_coords, timestamps

def min_max_normalize(data):
    min_val = min(data)
    max_val = max(data)
    normalized = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized

def calculate_relative_times(timestamps):
    start_time = timestamps[0]
    relative_times = [(t - start_time) for t in timestamps]
    return relative_times

def process_file(file_path):
    x_coords, y_coords, timestamps = read_data(file_path)
    
    normalized_x = min_max_normalize(x_coords)
    normalized_y = min_max_normalize(y_coords)
    relative_times = calculate_relative_times(timestamps)
    
    return normalized_x, normalized_y, relative_times

for i in range(1, 4):
    file_path = f'mouse_tracking/sample/flicks/{i}.txt'
    normalized_x, normalized_y, relative_times = process_file(file_path)
    # plot out the normalized data x-coordinates vs time_gap
    plt.plot(normalized_x, relative_times, marker='o', color='r', ls='')
    plt.savefig(f'mouse_tracking/flick_analysis/{i}_x_to_time.png')
    plt.close()
    
    # plot out the normalized data time_gap vs y-coordinates 
    plt.plot(relative_times, normalized_y, marker='o', color='r', ls='')
    plt.savefig(f'mouse_tracking/flick_analysis/{i}_time_to_y.png')
    plt.close()

