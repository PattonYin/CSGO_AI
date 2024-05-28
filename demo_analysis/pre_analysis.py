from data_preperator import Processor
import pickle
from plot_data import *

columns_to_checkout = ['life_state', 'move_collide', 'move_type', 'next_attack_time', 'ducking', 'spotted', 'is_walking', 'is_defusing', 'flash_duration', 'is_strafing']

columns_to_pick = ["health", "is_alive", "armor", "has_helmet", ]

params = dict(
    demo_path = "demo_analysis/demo/match730_003673760416913162325_1520768583_129.dem",
    player_name = "BERNARDO",
    columns = ["pitch", "yaw", "X", "Y", "Z", "shots_fired", "team_num"]
)

def plot_player_fire_data():
    
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
        print(f"Player:{tick_sample_all_data.iloc[i, 9]}, X:{tick_sample_all_data.iloc[i, 4]}, y:{tick_sample_all_data.iloc[i, 5]}")

    plot_tick(tick_sample_all_data, data_all=data)


def anime_player_movements(tick_rate=64):
    processor = Processor(params["demo_path"])
    data = processor.get_round_data(props=params['columns'])

    # processor.plot_attempt("BERNARDO")
    round_1_data = data[data['round_num'] == 1]
    plot_move(round_1_data, tick_rate=tick_rate)


def plot_all_player_movements():
    print(f"max x: {max(data['X'])}")
    print(f"min x: {min(data['X'])}")
    print(f"max y: {max(data['Y'])}")
    print(f"min y: {min(data['Y'])}")
    plot_positions(data)

if __name__ == '__main__':
    # print(round_1_data.head())

    # plot_player_fire_data()
    anime_player_movements(64)
    # plot_all_player_movements()
