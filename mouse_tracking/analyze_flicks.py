import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    
def flict_plot_0():
    plt.figure(figsize=(8, 8))
    colors = ['r', 'g', 'b']
    for i in range(1, 4):
        file_path = f'mouse_tracking/sample/flicks/{i}.txt'
        normalized_x, normalized_y, relative_times = process_file(file_path)
        speed_x = [normalized_x[i] - normalized_x[i-1] for i in range(1, len(normalized_x))]
        # plt.plot(speed_x, relative_times[1:], marker='o', color='r', ls='')
        
        def plot_flick():
            # plot out the normalized data x-coordinates vs time_gap
            plt.plot(relative_times, normalized_x, marker='o', color=colors[i-1], ls='')
            # plt.savefig(f'mouse_tracking/flick_analysis/{i}_x_to_time.png')
            
            # plot out the normalized data time_gap vs y-coordinates 
            # plt.plot(relative_times, normalized_y, marker='o', color='r', ls='')
            # plt.savefig(f'mouse_tracking/flick_analysis/{i}_time_to_y.png')
            # plt.show()
        plot_flick()

    plt.show()
    
def analyze_flick_trajectory(file_path):
    """involves separating the flicks into lists of x, y, and timestamps

    Args:
        file_path (string): input path
    """
    x_coords_list = []
    y_coords_list = []
    timestamps_list = []
    x_coords = []
    y_coords = []
    timestamps = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 4 and parts[1] == "Key.f2":
                if parts[2] == "released":
                    x_coords_list.append(x_coords)
                    y_coords_list.append(y_coords)
                    timestamps_list.append(timestamps)
                if parts[2] == "pressed":
                    x_coords = []
                    y_coords = []
                    timestamps = []
            if len(parts) == 4 and parts[0] == 'mouse':
                x_coords.append(float(parts[1]))
                y_coords.append(float(parts[2]))
                timestamps.append(float(parts[3]))

    # Normalize the data
    threshold_lower = 15
    threshold_upper = 100
    output_x, output_y, output_timestamps = [], [], []
    for x_coords, y_coords, timestamps in zip(x_coords_list, y_coords_list, timestamps_list):
        if len(x_coords) < threshold_lower or len(x_coords) > threshold_upper:
            continue
        output_x.append(min_max_normalize(x_coords))
        output_y.append(min_max_normalize(y_coords))
        output_timestamps.append(calculate_relative_times(timestamps))
    
    # Debug
    # for i in range(len(output_x)):
    #     print(f"length of x: {len(output_x[i])}, length of y: {len(output_y[i])}, length of timestamps: {len(output_timestamps[i])}")
    
    # Plotting            
    colors = ['r', 'g', 'b']
    for i in range(1,4):
        plt.plot(output_timestamps[i], output_x[i], marker='o', color=colors[i-1], ls='')
        plt.show()
    
    # Animation
    fig, ax = plt.subplots()
ax.set_xlim(0, 10)  # Set these limits according to your data
ax.set_ylim(0, 20)

line, = ax.plot([], [], 'r-', animated=True)

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data(flicks[frame]['x'], flicks[frame]['y'])
    return line,

ani = FuncAnimation(fig, update, frames=range(len(flicks)), init_func=init, blit=True, repeat=True)

plt.show()
    
if __name__ == "__main__":
    file_path = "data/aim_training/1/mouse_keyboard/mouse_keyboard_input.txt"
    analyze_flick_trajectory(file_path)
    