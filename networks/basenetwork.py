import torch
# import torch.nn.functional as F
# import pickle
# import os

class BaseNetwork(torch.nn.Module):
    def __init__(self, opt):
        super(BaseNetwork, self).__init__()
        self.name = "BaseNetwork"
        self._opt = opt

    def forward(self, input):
        raise NotImplementedError('forward not implemented for BaseNetwork')

    def _initialize_weight(self):
        raise NotImplementedError('initialize_weight not implemented')

#
