from models import Mutual_Information_Analysis

data_path = r"intermediate_data\model_training\10"

MI = Mutual_Information_Analysis()
MI.prepare_data(data_path=data_path)
X = MI.X