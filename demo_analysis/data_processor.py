from demoparser2 import DemoParser
import pandas as pd
import matplotlib.pyplot as plt

class Processor:
    def __init__(self, path="demo_analysis/demo/furia-vs-lynn-vision-nuke.dem"):
        self.parser = DemoParser(path)

    def get_player_data(self):
        df_player = self.parser.parse_ticks(wanted_props=["X", "Y", "Z", "pitch", "yaw"])
        return df_player
    
    def plot_attempt(self, player_name="z4KR"):
        df_player = self.get_player_data()
        df_player = df_player[df_player["name"] == player_name]
        df_player = df_player[:]
        print(df_player.head(20))
        # plot out in dots, with X as x-axis and Y as y-axis
        plt.scatter(df_player["X"], df_player["Y"], s=1)
        plt.show()
        
    
if __name__ == "__main__":
    match_01 = {
        "demo_path": "demo_analysis/demo/furia-vs-lynn-vision-nuke.dem",
        "player_name": "z4KR"
    }
    match_02 = {
        "demo_path": "demo_analysis/demo/match730_003673760416913162325_1520768583_129.dem",
        "player_name": "Aerith"
    }

    processor = Processor(match_02["demo_path"])
    # df_player = processor.get_player_data()
    # print(len(df_player)/(640*60))
    processor.plot_attempt(match_02["player_name"])
