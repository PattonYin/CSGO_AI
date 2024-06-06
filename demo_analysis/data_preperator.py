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



if __name__ == "__main__":
    demo_path = "demo_analysis/demo/match730_003673760416913162325_1520768583_129.dem"
    processor = Processor(demo_path)
    death_df = processor.query_death_df()
    win_rate = processor.win_rate_of_fights([death_df])
