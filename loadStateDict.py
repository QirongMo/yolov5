

from models.yolo import Model
import yaml
import pickle as pkl
import torch
from datetime import datetime
from copy import deepcopy
from utils.torch_utils import de_parallel

cfg = "data/cfg/yolo11/yolo11n.yaml"
nc = 80
model = Model(cfg, ch=3, nc=nc)  # create

old_weights = "/home/mqr/Desktop/ultralytics-main/yolo11n_state_dict.pkl"
with open(old_weights, 'rb') as f:
    state_dict = pkl.load(f)
# parial_dict = {}
# for i, (key, value) in enumerate(state_dict.items()):
#     if i >= 20:
#         break
#     parial_dict[key] = value

model.model.load_state_dict(state_dict, strict=False)

data = {
    "epoch": None,
    "best_fitness": None,
    "model": deepcopy(de_parallel(model)).half(),
    "ema": None,
    "updates": None,
    "optimizer": None,
    "opt": None,
    "date": datetime.now().isoformat(),
}
torch.save(data, "./yolo11n.pt")


