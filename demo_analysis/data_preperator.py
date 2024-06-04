from demoparser2 import DemoParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

from config import *

class Processor:
    def __init__(self, path="demo_analysis/demo/match730_003673760416913162325_1520768583_129.dem"):
        self.parser = DemoParser(path)
        self.tick_rate = 64
    
    def get_round_ticks(self):
        """Returns the start ticks and end ticks for a game
        
        Returns:
        start_ticks (list): list of start ticks for each round
        end_ticks (list): list of end ticks for each round
        """

        round_data = self.parser.parse_events(event_name=["round_start", "round_end"])
        if round_data[0][0] == 'round_start':
            start_df = round_data[0][1]
            end_df = round_data[1][1]
        else:
            start_df = round_data[1][1]
            end_df = round_data[0][1]

        start_ticks = start_df['tick'].values.tolist()
        end_ticks = end_df['tick'].values.tolist()
        return start_ticks, end_ticks
    
    def get_round_data(self, props):
        """With wanted columns to query given, return desired per-tick data with round_num as an additional column. This will return all ticks in the game

        Args:
            props (list): list of columns to be queried
            
        Returns:
            df_player (pd.DataFrame): dataframe of player data with round_num as an additional column
        """
        start_ticks, end_ticks = self.get_round_ticks()
        df_player = self.parser.parse_ticks(wanted_props=props)
        column_names = list(df_player.columns) + ["round_num"]
        output_df = pd.DataFrame(columns=column_names)
        # Filter out the data for each round
        for i in range(1, len(start_ticks)+1):
            round_df = df_player[(df_player["tick"] >= start_ticks[i-1]) & (df_player["tick"] <= end_ticks[i-1])]
            round_df = round_df.copy() 
            round_df["round_num"] = i
            output_df = pd.concat([output_df, round_df], ignore_index=True)
        
        return output_df
    
    def get_ticks(self, start_tick, end_tick, props):
        all_ticks = self.parser.parse_ticks(wanted_props=props)
        return all_ticks[(all_ticks["tick"] >= start_tick) & (all_ticks["tick"] <= end_tick)]
    
    def query_death_df(self):
        """
        ['assistedflash', 'assister_X', 'assister_Y', 'assister_name',
       'assister_steamid', 'attacker_X', 'attacker_Y', 'attacker_name',
       'attacker_steamid', 'attackerblind', 'distance', 'dmg_armor',
       'dmg_health', 'dominated', 'headshot', 'hitgroup', 'noreplay',
       'noscope', 'penetrated', 'revenge', 'thrusmoke', 'tick', 'user_X',
       'user_Y', 'user_name', 'user_steamid', 'weapon', 'weapon_fauxitemid',
       'weapon_itemid', 'weapon_originalowner_xuid', 'wipe']

        Returns:
            pd.DataFrame: a DataFrame containing the data of player_death event
        """
        death_df = self.parser.parse_event("player_death", player=["X", "Y"])
        return death_df
    
    def query_weapon_fire_df(self):
        """
        ['silenced', 'tick', 'user_X', 'user_Y', 'user_name', 'user_steamid',
       'weapon']

        Returns:
            pd.DataFrame: a DataFrame containing the data of weapon_fire event
        """
        weapon_fire_df = self.parser.parse_event("weapon_fire", player=["X", "Y"])
        return weapon_fire_df
    
    def query_blind_df(self):
        """
        ['attacker_X', 'attacker_Y', 'attacker_name', 'attacker_steamid',
       'blind_duration', 'entityid', 'tick', 'user_X', 'user_Y', 'user_name',
       'user_steamid']
       
        Returns:
            pd.DataFrame: a DataFrame containing the data of blind event
        """
        pass
    
    def query_ticks_df(self):
        pass

class Gun_Fight_Analysis(Processor):        
    def generate_pixel_map(self, side_length=50):
        """
        Generates a pixel map of the game area
        
        Args:
            side_length (int): the side length of the pixel map
            
        Returns:
            pixel_map (dict): a dictionary of the pixel map
        """
        x_coords = list(range(min_x, max_x+1, side_length))
        y_coords = list(range(min_y, max_y+1, side_length))
        square_ids = [(x, y) for x in x_coords for y in y_coords]
        square_dict = {square_id: 0 for square_id in square_ids}
        return square_dict
    
    def win_rate_of_fights(self, datas, area_togo=None):
        """
        Computes the win rate of gunfights on the basis of location of attacker on the map
        
        How to determine a fight:
        Factors considered:
        - A player dies
        
        Factors not yet considered:
        - HP of players
        - Weapons of players
        - Blind or not
        - Smoked or not

        The unit of location:
        square of size 50x50

        Possible_nextStep:
        Use NN to learn the gunfight win rate on the continuous level

        Args:
            datas (list): list of dataframes to be processed
                the dataframe has to be the output from 
                    `parser.parse_event("player_death")`
            area_togo: e.g. ((min_x, max_x), (min_y, max_y))
        """
        side_length = 100
        fight_count = self.generate_pixel_map(side_length=side_length)
        win_count = self.generate_pixel_map(side_length=side_length)
        square_ids = fight_count.keys()
        
        for data in datas:
            for index in range(len(data)):
                if (data.iloc[index]['attackerblind'] == True 
                    or data.iloc[index]['thrusmoke'] == True):
                    continue
                
                if area_togo is not None:
                    att_true_x = data.iloc[index]['attacker_X']
                    att_true_y = data.iloc[index]['attacker_Y']
                    dead_true_x = data.iloc[index]['user_X']
                    dead_true_y = data.iloc[index]['user_Y']
                    
                    if att_true_x < area_togo[0][0] or att_true_x > area_togo[0][1] or att_true_y < area_togo[1][0] or att_true_y > area_togo[1][1]:
                        continue
                    if dead_true_x < area_togo[0][0] or dead_true_x > area_togo[0][1] or dead_true_y < area_togo[1][0] or dead_true_y > area_togo[1][1]:
                        continue
                    
                attacker_x = (att_true_x // side_length * side_length)
                attacker_y = (att_true_y // side_length * side_length)

                dead_x = (dead_true_x // side_length * side_length)
                dead_y = (dead_true_y // side_length * side_length)
                
                try:
                    attacker_coord = (int(attacker_x), int(attacker_y))
                    dead_coord = (int(dead_x), int(dead_y))
                except ValueError:
                    # print(attacker_x, attacker_y)
                    # print(dead_x, dead_y)
                    continue    
                
                if attacker_coord not in square_ids:
                    print(data.iloc[index]['attacker_X'])
                    print(data.iloc[index]['attacker_Y'])
                
                
                win_count[attacker_coord] += 1
                fight_count[attacker_coord] += 1
                fight_count[dead_coord] += 1
                
        win_rate = {}
        
        for square_id in square_ids:
            if fight_count[square_id] == 0:
                win_rate[square_id] = None
            else:
                win_rate[square_id] = win_count[square_id]/fight_count[square_id]
                
        return win_rate
    
    def num_death_on_map(self, datas, area_togo=None):
        side_length = 100
        
        dead_count = self.generate_pixel_map(side_length=side_length)
        square_ids = dead_count.keys()
        
        for data in datas:
            for index in range(len(data)):
                # if (data.iloc[index]['attackerblind'] == True 
                #     or data.iloc[index]['thrusmoke'] == True):
                #     continue
                dead_true_x = data.iloc[index]['user_X']
                dead_true_y = data.iloc[index]['user_Y']
                
                if area_togo is not None:                    
                    if dead_true_x < area_togo[0][0] or dead_true_x > area_togo[0][1] or dead_true_y < area_togo[1][0] or dead_true_y > area_togo[1][1]:
                        continue

                dead_x = (dead_true_x // side_length * side_length)
                dead_y = (dead_true_y // side_length * side_length)
                
                try:
                    dead_coord = (int(dead_x), int(dead_y))
                except ValueError:
                    continue                   

                dead_count[dead_coord] += 1
                
        return dead_count
                
                
class Aim_Analysis(Processor):
    def aim_speed_case_study(self):
        # Constants
        sample_idx = 20
        tick_range = 100 # 100 ticks before the death event
        dyaw_lim = (-500, 500)
        dpitch_lim = (-1, 1)
        props_to_query = ["pitch", "yaw", "X", "Y", "Z"]
        
        dval = True
        
        # step 1. Find an event of death
        data = self.query_death_df()
        
        attacker_name = data.iloc[sample_idx]['attacker_name']
        attackee_name = data.iloc[sample_idx]['user_name']
        tick_sample = data.iloc[sample_idx]['tick']
        
        ticks = self.get_ticks(tick_sample-tick_range, tick_sample, props_to_query)
        
        fire_df = self.query_weapon_fire_df()
        fire_df = fire_df[(fire_df["tick"] >= tick_sample-tick_range) & (fire_df["tick"] <= tick_sample)]
        attacker_fire_df = fire_df[fire_df['user_name'] == f'{attacker_name}']
        attackee_fire_df = fire_df[fire_df['user_name'] == f'{attackee_name}']
        
        attacker_fire_ticks = attacker_fire_df.tick.tolist()
        attackee_fire_ticks = attackee_fire_df.tick.tolist()
                
        # step 2. Compute the dyaw/dt and dpitch/dt
        attacker_ticks = ticks[ticks['name'] == f'{attacker_name}']
        attackee_ticks = ticks[ticks['name'] == f'{attackee_name}']
    
        attacker_yaw_list = attacker_ticks.yaw.tolist()
        attacker_pitch_list = attacker_ticks.pitch.tolist()
        attackee_yaw_list = attackee_ticks.yaw.tolist()
        attackee_pitch_list = attackee_ticks.pitch.tolist()
        
        attacker_dyaw_list = np.diff(attacker_yaw_list)
        attacker_dpitch_list = np.diff(attacker_pitch_list)
        attackee_dyaw_list = np.diff(attackee_yaw_list)
        attackee_dpitch_list = np.diff(attackee_pitch_list)

        # step 3. Plot out the dyaw/dt and dpitch/dt
        # Highlight fire ticks
        def highlight_ticks(ax, data_list, fire_ticks, title, lim):
            ax.set_ylim(lim)
            ax.plot(data_list)
            ax.scatter([i+100-tick_sample for i in fire_ticks], [data_list[i+100-tick_sample] for i in fire_ticks], color='red', s=40)
            ax.set_title(title)
        
        # step 3. Plot out the dyaw/dt and dpitch/dt
        if dval:
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            highlight_ticks(ax[0, 0], attacker_dyaw_list, attacker_fire_ticks, "attacker_dyaw_list", dyaw_lim)
            highlight_ticks(ax[0, 1], attacker_dpitch_list, attacker_fire_ticks, "attacker_dpitch_list", dpitch_lim)
            highlight_ticks(ax[1, 0], attackee_dyaw_list, attackee_fire_ticks, "attackee_dyaw_list", dyaw_lim)
            highlight_ticks(ax[1, 1], attackee_dpitch_list, attackee_fire_ticks, "attackee_dpitch_list", dpitch_lim)
            plt.tight_layout()
            plt.show()
        else:
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            highlight_ticks(ax[0, 0], attacker_yaw_list, attacker_fire_ticks, "attacker_yaw_list")
            highlight_ticks(ax[0, 1], attacker_pitch_list, attacker_fire_ticks, "attacker_pitch_list")
            highlight_ticks(ax[1, 0], attackee_yaw_list, attackee_fire_ticks, "attackee_yaw_list")
            highlight_ticks(ax[1, 1], attackee_pitch_list, attackee_fire_ticks, "attackee_pitch_list")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    demo_path = "demo_analysis/demo/match730_003673760416913162325_1520768583_129.dem"
    processor = Processor(demo_path)
    death_df = processor.query_death_df()
    win_rate = processor.win_rate_of_fights([death_df])
