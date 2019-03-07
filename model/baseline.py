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
        if self._opt.load_epoch_C > 0 or self._opt.load_epoch_C == -1:
            self.load(self._opt.load_epoch_C)


    def forward(self, input):
        if not self._is_train:
            return self._classifier.forward(input)
        else:
            raise ValueError('model.forward is not for training')

    def update_parameters(self, batch_data, idx, print_flag=False, debug_flag=False):
        if self._is_train:
            # forward
            y_pred = self._classifier.forward(batch_data[:, :-1])
            Loss = self._lossfun(y_pred, batch_data[:, -1])

            # update accumulated loss
            x = float(self._opt.batch_size)
            self._eta = self._eta / (1. + x*self._eta)
            self._accum_loss = (1. - x*self._eta) * self._accum_loss + self._eta * Loss

            # display and debug
            if print_flag:
                print(" -> accumulated loss is %.8f" % self._accum_loss, flush=True)
                if debug_flag:
                    fname = "debug_{}_{}_iter.txt".format(str(self._classifier.name), str(idx))
                    debug_path = os.path.join(self._opt.debug_dir, fname)
                    self._classifier.weight_debug(debug_path)

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
            self._classifier = self._net_dict[self._opt.classifier_net](self._opt)
        except:
            print(self._opt.classifier_net)
            raise ValueError('network name not supported')

    def _set_optimizer(self):
        # self._optimizer = torch.optim.SGD(self._classifier.parameters(), lr=self._opt.lr_C)
        self._optimizer = torch.optim.Adam(self._classifier.parameters(), lr=self._opt.lr_C)

    def _set_lossfunction(self):
        self._lossfun = nn.CrossEntropyLoss()

    def save(self, epoch_idx):
        name = "{}_{}".format(self._classifier.name, self._opt.tag)
        self._save_network(self._classifier, name, epoch_idx)
        self._save_optimizer(self._optimizer, name, epoch_idx)

    def load(self, epoch_idx):
        name = "{}_{}".format(self._classifier.name, self._opt.tag)
        self._load_network(self._classifier, name, epoch_idx)
        self._load_optimizer(self._optimizer, name, epoch_idx)
