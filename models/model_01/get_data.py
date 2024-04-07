from neural_net_models.data_preperator import Processor

input_variables = ["X", "Y", "Z", "pitch", "yaw", "weapon", "ducking", "spotted", "armor_value", "is_walking"]
matches = ["demo_analysis/demo/furia-vs-lynn-vision-nuke.dem"]


for match in matches:
    processor = Processor(match)
    data = processor.get_round_data(props=input_variables)
    
