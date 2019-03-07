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
        self._balanced_datasize = 0

        # prepare training dataset
        self._dataset = load_training_data(self._opt)
        if self._opt.valid_num > 0:
            # self._dataset = self._dataset[:-self._opt.valid_num, :]
            self._dataset = self._dataset[:self._opt.valid_num, :] #for debug only
        self._training_size = self._dataset.shape[0]

        # Initialize data loader
        self._dataloader = CustomDataLoader(self._dataset, self._opt)

        # set model and start epoch for training, may load param within
        self._model = BaselineModel(self._opt)
        self._start_epoch = self._model.model_epoch

        # train model
        self._train()

    def _train(self):
        print("\nStart training model for: \n%s" % (str(self._model._classifier)))
        print("initial learning rate: %s" % self._opt.lr_C)
        # condition for continued training
        if self._start_epoch > 0:
            assert self._start_epoch < self._opt.max_epoch_C, \
                   "loaded epoch >= max_epoch_C, nothing to train"

        # train in normal way; i.e. not use triplet loss
        assert not self._opt.is_triplet, "not for training triplet model"
        if self._opt.is_balanced:
            self._data_train, datasize = self._dataloader.load_balanced()
            self._balanced_datasize = datasize
        else:
            self._data_train = self._dataloader.load_batches()

        # Main loop, train each epoch
        for i_epoch in range(self._start_epoch+1, self._opt.max_epoch_C+1):
            epoch_time_start = time.time()

            # train epoch
            print("\nStart epoch %d / %d, \t at %s" % \
                  (i_epoch, self._opt.max_epoch_C, time.asctime()))
            self._train_epoch(i_epoch)

            # save model after each epoch here, note i_epoch is 0-based
            if (i_epoch) % self._opt.save_freq == 0 or (i_epoch) == self._opt.max_epoch_C:
                self._model.save(i_epoch)

            # training time for each epoch
            time_cost = time.time() - epoch_time_start
            print("End of epoch %d / %d \t Time taken: %d sec (or % d min)" % \
                  (i_epoch, self._opt.max_epoch_C, time_cost, time_cost / 60.))

            # updata learning rate
            # if i_epoch > ...

        # end of training
        print("", flush=True)

    def _train_epoch(self, i_epoch):
        # display records for the number of trained iters
        loss_check_count = 0
        if self._opt.is_balanced:
            total_iter = self._balanced_datasize
        else:
            total_iter = self._training_size

        loss_check_freq = self._opt.loss_check_freq
        if loss_check_freq == -1:
            loss_check_freq = int(total_iter / self._opt.max_loss_check)

        for i_train_batch, train_batch in enumerate(self._data_train):

            # update & display progress
            num_iter = (i_train_batch+1)*self._opt.batch_size
            print_flag = (int(num_iter/loss_check_freq) != loss_check_count) \
                         and (loss_check_count < self._opt.max_loss_check)
            if print_flag:
                print("  [%6.2f%s,  %7d / %7d]" % \
                      (float(num_iter)/float(total_iter)*100., "%", num_iter, total_iter), end='')
                loss_check_count += 1

            self._model.update_parameters(train_batch, num_iter, print_flag, self._opt.is_debug)

        # end of display progress
        print(" done", flush=True)

    def _display_progress(self, num_iter, total_iter):
        pass


if __name__ == '__main__':
    Train()
#