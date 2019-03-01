import torch
import numpy as np
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

    def weight_debug(self, debug_path):

        with open(debug_path, 'a') as f:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    f.write(name+"\n")
                    np.savetxt(f, param.detach().numpy(), delimiter=',', fmt='%12.5f')
                    f.write("\n")




#
