

import pickle
from ultralytics import YOLO

# 加载模型
model = YOLO("./yolo11n.pt")

state_dict: dict = model.model.model.state_dict()
with open("yolo11n_state_dict.pkl", 'wb') as f:
    pickle.dump(state_dict, f)