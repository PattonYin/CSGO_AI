from demoparser2 import DemoParser
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

from config import *

class Processor_Core:
    def __init__(self, path="demo_analysis/demo/match730_003673760416913162325_1520768583_129.dem"):
        self.parser = DemoParser(path)
        self.tick_rate = 64
    
class Processor_Efficient(Processor_Core):
    """This is designed for model training

    """
    def __init__(self, path="demo_analysis/demo/match730_003673760416913162325_1520768583_129.dem", tick_range=16*3):
        super().__init__(path)
        self.tick_range = tick_range
        
    def ticks_for_training(self, fixed_range=False):
        """Collects the ticks for training the model

        Returns:
            tick_ranges (list (tuple)): list of tick ranges for training
            attacker (list (str)): list of attackers
            attackees (list (str)): list of attackees
        """
        death_df = self.parser.parse_event("player_death")
        death_ticks = death_df['tick'].values.tolist()
        attackers = death_df['attacker_name'].values.tolist()
        attackees = death_df['user_name'].values.tolist()
        tick_ranges = []
        
        if fixed_range:
            for death_tick in death_ticks:
                tick_ranges.append((death_tick - self.tick_range, death_tick))
        else:
            weapon_fire_df = self.parser.parse_event("weapon_fire")

            for index, death_tick in enumerate(death_ticks):
                fire_df = weapon_fire_df[(weapon_fire_df['tick'] >= death_tick - self.tick_range) & (weapon_fire_df['tick'] <= death_tick)]
                fire_df = fire_df[(fire_df['user_name'] == attackees[index]) | (fire_df['user_name'] == attackers[index])]

                if len(fire_df) == 0:
                    continue
                fire_tick = fire_df['tick'].values.tolist()[0]
                tick_ranges.append((fire_tick, death_tick))
                
        print("tick_ranges ready")
        return tick_ranges, attackers, attackees

        

        
    
class Processor(Processor_Core):
    """This is designed for data analysis

    """
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
    
    def query_bomb_plant_info(self):
        """Returns the information associated with the bomb

        Returns:
            _type_: _description_
        """
        bomb_df = self.parser.parse_event('bomb_planted')
        return bomb_df
    
    def query_flash_info(self):
        """
        ['entityid', 'tick', 'user_X', 'user_Y', 'user_Z', 'user_name',
       'user_steamid', 'x', 'y', 'z']
        """
        flash_df = self.parser.parse_event('flashbang_detonate', player=["team_num"])
        return flash_df
    
    def query_smoke_info(self):
        """
        ['entityid', 'tick', 'user_X', 'user_Y', 'user_Z', 'user_name',
       'user_steamid', 'x', 'y', 'z']
        """
        smoke_df = self.parser.parse_event('smokegrenade_detonate', player=["team_num"])
        return smoke_df
        
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
    
    def generate_pixel_map_to_pixel_map(self, side_length=50):
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


    def data_to_pixel_map(self, data, side_length=50):
        """
        Converts the data to pixel map
        
        Args:
            data (pd.DataFrame): the data to be converted to pixel map
            side_length (int): the side length of the pixel map
            
        Returns:
            pixel_map (dict): a dictionary of the pixel map
        """
        pixel_map = self.generate_pixel_map(side_length)
        for index, row in data.iterrows():
            x = row['x']
            y = row['y']
            x = x - (x % side_length)
            y = y - (y % side_length)
            pixel_map[(x, y)] += 1
        
        return pixel_map

if __name__ == "__main__":
    demo_path = "demo_analysis/demo/match730_003673760416913162325_1520768583_129.dem"
    processor = Processor(demo_path)
    death_df = processor.query_death_df()
    win_rate = processor.win_rate_of_fights([death_df])
