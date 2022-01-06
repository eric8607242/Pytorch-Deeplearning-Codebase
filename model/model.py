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

        self._initialize_weights()

    def forward(self, x):
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, (nn.Linear, nn.Embedding)):
                m.weight.data.normal_(0, 0.02)
