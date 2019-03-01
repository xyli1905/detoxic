# for model training
from options.base_options import BaseOption
from model.baseline import BaselineModel
from utils.dataloader import CustomDataLoader, load_training_data
import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle
import time
import os


'''
NOTE the last two parameters are added for easy use in jupyter notebook
will be removed in later versions
'''
class Train:
    def __init__(self):
        # parse param & options & setup
        self._opt = BaseOption().parse()

        # prepare training dataset
        self._dataset = load_training_data(self._opt)
        if self._opt.valid_num > 0:
            self._dataset = self._dataset[:-self._opt.valid_num, :]
        self._training_size = self._dataset.shape[0]

        # Initialize data loader
        self._dataloader = CustomDataLoader(self._dataset, self._opt)

        # set model and start epoch for training, may load param within
        self._model = BaselineModel(self._opt)
        self._start_epoch = self._model.model_epoch

        # train model
        self._train()

    def _train(self):
        print("\nStart training model for: \n%s" % (str(self._model._network)))
        # condition for continued training
        if self._start_epoch > 0:
            assert self._start_epoch < self._opt.max_epoch, \
                   "loaded epoch >= max_epoch, nothing to train"

        # if train in normal way; i.e. not use triplet loss
        if not self._opt.triplet:
            self._data_train = self._dataloader.load_batches()

        # Main loop, train each epoch
        for i_epoch in range(self._start_epoch+1, self._opt.max_epoch+1):
            epoch_time_start = time.time()

            # if use triplet loss re-sample triplet for every epoch
            if self._opt.triplet:
                self._data_train = self._dataloader.load_triplets()

            # train epoch
            print("\nStart epoch %d / %d, \t at %s" % \
                  (i_epoch, self._opt.max_epoch, time.asctime()))
            self._train_epoch(i_epoch)

            # save model after each epoch here, note i_epoch is 0-based
            if (i_epoch) % self._opt.save_freq == 0 or (i_epoch) == self._opt.max_epoch:
                self._model.save(i_epoch)

            # training time for each epoch
            time_cost = time.time() - epoch_time_start
            print("End of epoch %d / %d \t Time taken: %d sec (or % d min)" % \
                  (i_epoch, self._opt.max_epoch, time_cost, time_cost / 60.))

            # updata learning rate
            # if i_epoch > ...

    def _train_epoch(self, i_epoch):
        # display records for the number of trained iters
        print(" 0%s"%("%"), end='')
        p = 0.1

        for i_train_batch, train_batch in enumerate(self._data_train):

            self._model.update_parameters(train_batch)

            # update & display progress
            if (i_train_batch+1)*self._opt.batch_size >= p*self._training_size:
                print(" - %.0f%s" %(100.*p,"%"), end='')
                p += 0.1

        # end of display progress
        print(" done")

    def _display_progress(self):
        pass


if __name__ == '__main__':
    Train()
#