# including Bow, EmbBoW, EmbLR, LSTM & GRU, trained in normal way
from .basemodel import BaseModel
import torch
import torch.nn as nn
import numpy as np
import os
import pickle


class BaselineModel(BaseModel):
    def __init__(self, opt):
        super(BaselineModel, self).__init__(opt)
        self.name = "Baseline Model"

        # create the model, including networks, lossfunction and optimizer
        self._create_model()

        # check if need load previous parameters
        if self._opt.load_epoch_idx > 0 or self._opt.load_epoch_idx == -1:
            self.load(self._opt.load_epoch_idx)
            self.model_epoch = self._opt.load_epoch_idx


    def forward(self, input):
        if not self._is_train:
            return self._network.forward(input)
        else:
            raise ValueError('model.forward is not for training')

    def update_parameters(self, batch_data, idx, print_flag=False, debug_flag=False):
        if self._is_train:
            # forward
            y_pred = self._network.forward(batch_data[:, :-1])
            Loss = self._lossfun(y_pred, batch_data[:, -1])

            # display and debug
            if print_flag:
                print(" -> current loss is %.5f" % Loss, flush=True)
                if debug_flag:
                    fname = "debug_{}_iter.txt".format(str(idx))
                    debug_path = os.path.join(self._opt.debug_dir, fname)
                    self._network.weight_debug(debug_path)

            # backprop
            self._optimizer.zero_grad()
            Loss.backward()
            self._optimizer.step()
        else:
            raise ValueError('model.update_parameters is for training only')


    def _create_model(self):
        self._set_network()
        self._set_optimizer()
        self._set_lossfunction()

    def _set_network(self):
        try:
            self._network = self._net_dict[self._opt.network_name](self._opt)
        except:
            raise ValueError('network name not supported')

    def _set_optimizer(self):
        self._optimizer = torch.optim.SGD(self._network.parameters(), lr=self._opt.lr)

    def _set_lossfunction(self):
        self._lossfun = nn.CrossEntropyLoss()

    def save(self, epoch_idx):
        name = self._network.name
        self._save_network(self._network, name, epoch_idx)
        self._save_optimizer(self._optimizer, name, epoch_idx)

    def load(self, epoch_idx):
        name = self._network.name
        self._load_network(self._network, name, epoch_idx)
        self._load_optimizer(self._optimizer, name, epoch_idx)
