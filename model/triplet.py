from networks.LR import EmbLR
from networks.linear import LinearClassifier
from .basemodel import BaseModel
import torch
import torch.nn as nn
import numpy as np
import os
import pickle


class TripletModel(BaseModel):
    def __init__(self, opt):
        super(TripletModel, self).__init__(opt)
        self.name = "Triplet Model"
        self.model_epoch_E = 0
        self.model_epoch_C = 0
        self._encoder_trainable = True

        # create the model, including networks, lossfunction and optimizer
        self._create_model()

        # check if need load previous parameters
        if self._opt.load_epoch_E > 0 or self._opt.load_epoch_E == -1:
            self.load(self._opt.load_epoch_E, "encoder")
            self.model_epoch_E = self.model_epoch
        if self._opt.load_epoch_C > 0 or self._opt.load_epoch_C == -1:
            self.load(self._opt.load_epoch_C, "classifier")
            self.model_epoch_C = self.model_epoch


    def forward(self, input):
        if not self._is_train:
            encoding = self._encoder.forward(input, use_encoding=True)
            out = self._classifier.forward(encoding)
            return out
        else:
            raise ValueError('model.forward is not for training')

    def update_encoder_parameters(self, batch_data, idx, print_flag=False, debug_flag=False):
        if self._is_train:
            # enable param.requires_grad in encoder
            if not self._encoder_trainable:
                self._encoder_trainable = True
                for name, param in self._encoder.named_parameters():
                    if name == "embed.weight":
                        param.requires_grad = self._opt.is_emb_trainable
                    else:
                        param.requires_grad = True

            # forward
            anchor = self._encoder.forward(batch_data["anchor"][:,:-1], use_encoding=True)
            positive = self._encoder.forward(batch_data["positive"][:,:-1], use_encoding=True)
            negative = self._encoder.forward(batch_data["negative"][:,:-1], use_encoding=True)
            Loss = self._encoder_lossfun(anchor, positive, negative)

            # update accumulated loss
            x = float(self._opt.batch_size)
            self._eta = self._eta / (1. + x*self._eta)
            self._accum_loss = (1. - x*self._eta) * self._accum_loss + self._eta * Loss

            # display and debug
            if print_flag:
                print(" -> accumulated loss is %.8f" % self._accum_loss, flush=True)
                if debug_flag:
                    fname = "debug_encoder_{}_{}_iter.txt".format(str(self._encoder.name), str(idx))
                    debug_path = os.path.join(self._opt.debug_dir, fname)
                    self._encoder.weight_debug(debug_path)

            # backward
            self._encoder_optimizer.zero_grad()
            Loss.backward()
            self._encoder_optimizer.step()
        else:
            raise ValueError('model.update_parameters is for training only')

    def update_classifier_parameters(self, batch_data, idx, print_flag=False, debug_flag=False):
        if self._is_train:
            # disable param.requires_grad in encoder
            if self._encoder_trainable:
                self._encoder_trainable = False
                for param in self._encoder.parameters():
                    param.requires_grad = False

            # forward
            encoding = self._encoder.forward(batch_data[:, :-1], use_encoding=True)
            y_pred = self._classifier.forward(encoding)
            Loss = self._classifier_lossfun(y_pred, batch_data[:, -1])

            # update accumulated loss
            x = float(self._opt.batch_size)
            self._eta = self._eta / (1. + x*self._eta)
            self._accum_loss = (1. - x*self._eta) * self._accum_loss + self._eta * Loss

            # display and debug
            if print_flag:
                print(" -> accumulated loss is %.8f" % self._accum_loss, flush=True)
                if debug_flag:
                    fname = "debug_classifier_{}_{}_iter.txt".format(str(self._classifier.name), str(idx))
                    debug_path = os.path.join(self._opt.debug_dir, fname)
                    self._classifier.weight_debug(debug_path)

            # backward
            self._classifier_optimizer.zero_grad()
            Loss.backward()
            self._classifier_optimizer.step()
        else:
            raise ValueError('model.update_parameters is for training only')


    def _create_model(self):
        self._set_network()
        self._set_optimizer()
        self._set_lossfunction()

    def _set_network(self):
        self._encoder = EmbLR(self._opt) #output dim = 64
        self._classifier = LinearClassifier(self._opt, size=(128, 64, 2))

    def _set_optimizer(self):
        self._encoder_optimizer = torch.optim.SGD(self._encoder.parameters(), lr=self._opt.lr_E)
        self._classifier_optimizer = torch.optim.SGD(self._classifier.parameters(), lr=self._opt.lr_C)

    def _set_lossfunction(self):
        self._encoder_lossfun = nn.TripletMarginLoss(margin=self._opt.margin, p=2)
        self._classifier_lossfun = torch.nn.CrossEntropyLoss()

    def save(self, epoch_idx, save_type):
        if save_type == "encoder":
            name = '{}_{}'.format("encoder", self._encoder.name)
            self._save_network(self._encoder, name, epoch_idx)
            self._save_optimizer(self._encoder_optimizer, name, epoch_idx)
        elif save_type == "classifier":
            name = '{}_{}'.format("classifier", self._classifier.name)
            self._save_network(self._classifier, name, epoch_idx)
            self._save_optimizer(self._classifier_optimizer, name, epoch_idx)

    def load(self, epoch_idx, load_type):
        if load_type == "encoder":
            name = '{}_{}'.format("encoder", self._encoder.name)
            self._load_network(self._encoder, name, epoch_idx)
            self._load_optimizer(self._encoder_optimizer, name, epoch_idx)
        elif load_type == "classifier":
            name = '{}_{}'.format("classifier", self._classifier.name)
            self._load_network(self._classifier, name, epoch_idx)
            self._load_optimizer(self._classifier_optimizer, name, epoch_idx)