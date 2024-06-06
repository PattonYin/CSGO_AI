from data_preperator import *
from analysis import *
import pickle
import os
from plot_module import *
from tqdm import tqdm
from config import *

columns_to_checkout = ['life_state', 'move_collide', 'move_type', 'next_attack_time', 'ducking', 'spotted', 'is_walking', 'is_defusing', 'flash_duration', 'is_strafing']

columns_to_pick = ["health", "is_alive", "armor", "has_helmet", ]

params = dict(
    demo_path = "demo_analysis/demo/match730_003673760416913162325_1520768583_129.dem",
    player_name = "BERNARDO",
    columns = ["pitch", "yaw", "X", "Y", "Z", "shots_fired", "team_num"]
)

def visualize_player_fire_data():
    """Plots the player's position and orientation at the fire tick
    """
    processor = Processor(params["demo_path"])
    data = processor.get_round_data(props=params['columns'])

    # processor.plot_attempt("BERNARDO")
    round_1_data = data[data['round_num'] == 1]
    # round_1_data_bernardo = round_1_data[round_1_data['name'] == 'BERNARDO']
    # round_1_data_bernardo_fired = round_1_data_bernardo[round_1_data_bernardo['shots_fired'] == 1]
    # ticks = round_1_data_bernardo_fired.tick.tolist()
    # print(ticks[:10])
    # tick_sample = ticks[0]
    # tick_sample_all_data = data[data['tick'] == tick_sample]

    round_1_data_anyone_fired = round_1_data[round_1_data['shots_fired'] == 1]
    ticks_anyone_fired = round_1_data_anyone_fired.tick.tolist()
    tick_sample = ticks_anyone_fired[128*4]
    print(f"at tick {tick_sample}")
    tick_sample_all_data = data[data['tick'] == tick_sample]
    print(tick_sample_all_data.columns)
    for i in range(len(tick_sample_all_data)):
        print(f"Player:{tick_sample_all_data.iloc[i, 9]}, X:{tick_sample_all_data.iloc[i, 4]}, y:{tick_sample_all_data.iloc[i, 5]}, p: {tick_sample_all_data.iloc[i, 2]}, y: {tick_sample_all_data.iloc[i, 3]}")

    plot_tick(tick_sample_all_data, data_all=data)

def anime_player_movements(tick_rate=64):
    """Animates the player movement in a round

    Args:
        tick_rate (int, optional): The frame rate of the animation. Defaults to 64.
    """
    processor = Processor(params["demo_path"])
    data = processor.get_round_data(props=params['columns'])

    # processor.plot_attempt("BERNARDO")
    round_1_data = data[data['round_num'] == 1]
    plot_move(round_1_data, tick_rate=tick_rate)


def visualize_all_player_movements():
    print(f"max x: {max(data['X'])}")
    print(f"min x: {min(data['X'])}")
    print(f"max y: {max(data['Y'])}")
    print(f"min y: {min(data['Y'])}")
    plot_positions(data)

def visualize_win_rate(folder_dir='demo_analysis/demo/de_dust2', area_todo=None):
    """
    Currently scraped May 28th and 27th demo data
    
    """
    demo_paths = os.listdir(folder_dir)
    death_dfs = []
    for demo in tqdm(demo_paths, desc="Processing demos"):    
        processor = Gun_Fight_Analysis(os.path.join(folder_dir, demo))
        death_df = processor.query_death_df()
        death_dfs.append(death_df)
    win_rate = processor.win_rate_of_fights(death_dfs, area_todo)
    plot_pixel_map(win_rate, 100, title="win_rate_01")
    

def visualize_gun_fight_histories(folder_dir, area_todo=None):
    demo_paths = os.listdir(folder_dir)[:5]
    death_dfs = []
    for demo in tqdm(demo_paths, desc="Processing demos"):    
        processor = Gun_Fight_Analysis(os.path.join(folder_dir, demo))
        death_df = processor.query_death_df()
        death_dfs.append(death_df)
    dead_count = processor.num_death_on_map(death_dfs, area_todo)
    plot_pixel_map(dead_count, 100, title="dead_count_01")
    
def case_study_aim_speed():
    demo_path_case = r"X:\code\CSGO_AI\demo_analysis\demo\demos_extracted\de_dust2\g2-vs-vitality-m1-dust2.dem"
    processor = Aim_Analysis_Case_Study(demo_path_case)
    processor.aim_speed()
    
def case_study_aim_accuracy():
    demo_path_case = r"X:\code\CSGO_AI\demo_analysis\demo\demos_extracted\de_dust2\g2-vs-vitality-m1-dust2.dem"
    processor = Aim_Analysis_Case_Study(demo_path_case)
    # processor.aim_accuracy()
    processor.aim_accuracy_individual()

def group_study_aim_accuracy():
    demo_path_case = r"X:\code\CSGO_AI\demo_analysis\demo\demos_extracted\de_dust2\g2-vs-vitality-m1-dust2.dem"
    processor = Aim_Analysis_Group(demo_path_case)
    processor.aim_accuracy_individual_plot()
    
def case_study_spray_control():
    demo_path_case = r"X:\code\CSGO_AI\demo_analysis\demo\demos_extracted\de_dust2\g2-vs-vitality-m1-dust2.dem"
    processor = Aim_Analysis_Case_Study(demo_path_case)
    processor.spray_control()

if __name__ == '__main__':
    # print(round_1_data.head())

    # visualize_player_fire_data()
    # anime_player_movements(64)
    # visualize_all_player_movements()
    
    # folder_dir = r"X:\code\CSGO_AI\demo_analysis\demo\demos_extracted\de_dust2"
    # area_todo = ((500, max_x), (200, 1500))
    # visualize_win_rate(folder_dir, area_todo)
    # visualize_gun_fight_histories(folder_dir)
    
    # case_study_aim_speed()
    # case_study_aim_accuracy()
    group_study_aim_accuracy()