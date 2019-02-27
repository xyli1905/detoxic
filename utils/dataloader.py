from options.base_options import BaseOption
import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle
import time
import os


class CustomDataLoader:
    def __init__(self, data, opt):
        # note data is torch.tensor
        self.name = 'customized_dataloader'
        self._opt = opt

        if self._opt.triplet:
            # anchor & positive from _data_pos; negtive from _data_neg
            self._data_pos = data[data[:,-1] == 1]
            self._data_neg = data[data[:,-1] == 0]
            self._pos_size = self._data_pos.shape[0]
            self._neg_size = self._data_neg.shape[0]
        else:
            # normal
            self._data = data

    def load_batches(self):
        return DataLoader(self._data, batch_size=self._opt.batch_size,
                          shuffle=True, drop_last=False)

    def load_triplets(self):
        assert self._opt.iter_size > self._opt.batch_size, \
               "iter_size too small (<batch_size)"

        # get index
        temp_idx = np.random.choice(self._pos_size,
                                    size=(self._opt.iter_size,2),
                                    # replace=False
                                   )
        negative_idx = np.random.choice(self._neg_size,
                                        size=(self._opt.iter_size,),
                                        # replace=False
                                       )
        anchor_idx = temp_idx[:,0]
        positive_idx = temp_idx[:,1]

        # fill the triplets
        triplet_batch = []

        num_batch = int(self._opt.iter_size / self._opt.batch_size)
        for i_batch in range(num_batch):

            idx_low = i_batch * self._opt.batch_size
            idx_high = (i_batch + 1) * self._opt.batch_size

            anchor = self._data_pos[anchor_idx[idx_low: idx_high]]

            positive = self._data_pos[positive_idx[idx_low: idx_high]]

            negative = self._data_neg[negative_idx[idx_low: idx_high]]

            # triplet = (anchor, positive, negative)
            triplet = {"anchor": anchor,
                       "positive": positive,
                       "negative": negative
                      }

            triplet_batch.append(triplet)

        # possible for last batch
        last_batch_size = self._opt.iter_size % self._opt.batch_size
        if last_batch_size != 0:

            idx_low = num_batch*self._opt.batch_size

            anchor = self._data_pos[anchor_idx[idx_low: ]]

            positive = self._data_pos[positive_idx[idx_low: ]]

            negative = self._data_neg[negative_idx[idx_low: ]]

            # triplet = (anchor, positive, negative)
            triplet = {"anchor": anchor,
                       "positive": positive,
                       "negative": negative
                      }

            triplet_batch.append(triplet)

        return triplet_batch
#





