from .basenetwork import BaseNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearClassifier(BaseNetwork):
    def __init__(self, opt, size):
        super(LinearClassifier, self).__init__(opt)
        self.name = "LinearC"

        # define networks
        self._layer_num = len(size) - 1
        layers = []
        for i in range(self._layer_num):
            layers.append(nn.Linear(size[i], size[i+1]))
        self.linearC = nn.Sequential(*layers)

    def forward(self, input):
        return self.linearC(input)