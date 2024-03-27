from demoparser2 import DemoParser
import pandas as pd

class Processor:
    def __init__(self, path="demo_analysis/demo/furia-vs-lynn-vision-nuke.dem"):
        self.parser = DemoParser(path)

    def get_player_data(self):
        df_player = self.parser.parse_ticks(wanted_props=["X", "Y", "Z", "pitch", "yaw"])
        return df_player
    
if __name__ == "__main__":
    demo_path = "demo_analysis/demo/furia-vs-lynn-vision-nuke.dem"
    processor = Processor(demo_path)
    df_player = processor.get_player_data()
    print(len(df_player)/(640*60))
    
