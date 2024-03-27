from awpy import Demo
import sys

demo_path = "demo/furia-vs-lynn-vision-nuke.dem"
dem = Demo(file=demo_path)

dem.header
dem.grenades
dem.kills
dem.damages
dem.bomb
dem.smokes
dem.infernos
dem.weapon_fires
dem.ticks