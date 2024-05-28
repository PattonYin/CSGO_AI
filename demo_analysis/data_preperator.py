from demoparser2 import DemoParser
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

class Processor:
    def __init__(self, path="demo_analysis/demo/furia-vs-lynn-vision-nuke.dem"):
        self.parser = DemoParser(path)

    def round_ticks(self):
        the_ticks = self.parser.parse_events(event_name=["round_start", "round_end"])
        
        end_df = the_ticks[0][1]
        start_df = the_ticks[1][1]
        # rename the tick column in start_df to start_tick
        start_df = start_df.rename(columns={"tick": "start_tick"})
        end_df = end_df.rename(columns={"tick": "end_tick"})
        end_df["round"] = end_df["round"] - 1
        # merge the two dataframes
        the_ticks = pd.merge(start_df, end_df, on="round", how="inner")
        return the_ticks

    def get_player_data(self, props=["X", "Y", "Z", "pitch", "yaw"]):
        df_player = self.parser.parse_ticks(wanted_props=props)
        return df_player
    
    def get_round_data(self, props=["X", "Y", "Z", "pitch", "yaw"]):
        df_ticks = self.round_ticks()
        df_player = self.get_player_data(props=props)
        total_df = df_player[(df_player["tick"] >= df_ticks["start_tick"][0]) & (df_player["tick"] <= df_ticks["end_tick"][0])].copy()
        total_df["round"] = 1
        for i in range(1, len(df_ticks)):
            round_df = df_player[(df_player["tick"] >= df_ticks["start_tick"][i]) & (df_player["tick"] <= df_ticks["end_tick"][i])].copy()
            round_df["round"] = i+1
            total_df = pd.concat([total_df, round_df])
        return total_df
    
    def plot_attempt(self, player_name="z4KR", round_num=list(range(1,10))):
        df_player = self.get_round_data()
        df_player = df_player[df_player["name"] == player_name]
        df_player = df_player[df_player["round"].isin(round_num)]
        print(df_player.head(20))
        # plot out in dots, with X as x-axis and Y as y-axis
        plt.scatter(df_player["X"], df_player["Y"], s=1)
        plt.show()
        
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
        """With wanted columns to query given, return desired per-tick data with round_num as an additional column

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
    
    
if __name__ == "__main__":
    match_01 = {
        "demo_path": "demo_analysis/demo/furia-vs-lynn-vision-nuke.dem",
        "player_name": "z4KR"
    }
    match_02 = {
        "demo_path": "demo_analysis/demo/match730_003673760416913162325_1520768583_129.dem",
        "player_name": "BERNARDO"
    }

    processor = Processor(match_02["demo_path"])
    # df_player = processor.get_player_data()
    # print(df_player.head(20))
    
    # print(len(df_player)/(640*60))
    # processor.plot_attempt(match_02["player_name"])
    data = processor.get_player_data(props=["armor_value", "direction"])
    # drop first 300 rows
    data = data[300:]
    data.to_csv("temp_01.csv")
    print(data.head(20))
    
    # ticks = processor.round_ticks()
    
    # print((processor.get_round_data()).head())
