import math 

import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, model_config, *args, **kwargs):
        super(Model, self).__init__()
        bn_track_running_stats = model_config["bn_track_running_stats"]
        bn_momentum = model_config["bn_momentum"]

        classes = model_config["classes"]

    def forward(self, x):
        return y
