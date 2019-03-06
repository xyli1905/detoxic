# for model training
from options.base_options import BaseOption
from model.triplet import TripletModel
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
        self._opt.is_triplet = True

        # prepare training dataset
        self._dataset = load_training_data(self._opt)
        if self._opt.valid_num > 0:
            # self._dataset = self._dataset[:-self._opt.valid_num, :]
            self._dataset = self._dataset[:self._opt.valid_num, :] # for debug only
        self._training_size = self._dataset.shape[0]

        # Initialize data loader
        self._dataloader = CustomDataLoader(self._dataset, self._opt)

        # set model and start epoch for training, may load param within
        self._model = TripletModel(self._opt)
        self._start_epoch_E = self._model.model_epoch_E
        self._start_epoch_C = self._model.model_epoch_C

        # train model
        self._train()

    def _train(self):
        print("\nStart training model for: \n%s\n%s" % (str(self._model._encoder), str(self._model._classifier)))
        # condition for continued training
        train_encoder = True
        train_classifier = True
        if self._start_epoch_E > 0:
            if self._start_epoch_E < self._opt.max_epoch_E:
                train_encoder = False
        if self._start_epoch_C > 0:
            if self._start_epoch_C < self._opt.max_epoch_C:
                train_classifier = False
        if not (train_encoder or train_classifier):
            raise ValueError("loaded epoch >= max_epoch, nothing to train")

        # Main loop, train each epoch
        for i in range(self._opt.round_num):
            print("\n====== Train for Round #%d / %d ======" % (i+1, self._opt.round_num))
            if train_encoder:
                print("\nTraining encoder ......")
                self._train_by_part(train_type='encoder')
            if train_classifier:
                print("\nTraining classifier ......")
                self._train_by_part(train_type='classifier')

        # end of training
        print("", flush=True)

    def _train_by_part(self, train_type):
        if train_type == "encoder":
            start_epoch = self._start_epoch_E
            max_epoch = self._opt.max_epoch_E

        if train_type == "classifier":
            self._data_train = self._dataloader.load_batches()
            start_epoch = self._start_epoch_C
            max_epoch = self._opt.max_epoch_C

        for i_epoch in range(start_epoch+1, max_epoch+1):
            epoch_time_start = time.time()

            # re-sample triplet for every epoch
            if train_type == "encoder":
                self._data_train = self._dataloader.load_triplets()

            # train epoch
            print("\nStart epoch %d / %d, \t at %s" % \
                  (i_epoch, max_epoch, time.asctime()))
            self._train_epoch(i_epoch, train_type)

            # save model after each epoch here, note i_epoch is 0-based
            if (i_epoch) % self._opt.save_freq == 0 or (i_epoch) == max_epoch:
                self._model.save(i_epoch, train_type)

            # training time for each epoch
            time_cost = time.time() - epoch_time_start
            print("End of epoch %d / %d \t Time taken: %d sec (or % d min)" % \
                  (i_epoch, max_epoch, time_cost, time_cost / 60.))
            
            # updata learning rate
            # if i_epoch > ...

    def _train_epoch(self, i_epoch, train_type):
        # display records for the number of trained iters
        loss_check_count = 0
        if  train_type == "encoder":
            total_iter = self._opt.iter_size
        elif train_type == "classifier":
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

            if train_type == "encoder":
                self._model.update_encoder_parameters(train_batch, num_iter, print_flag, self._opt.is_debug)
            elif train_type == "classifier":
                self._model.update_classifier_parameters(train_batch, num_iter, print_flag, self._opt.is_debug)

        # end of display progress
        print(" done", flush=True)

    def _display_progress(self, num_iter, total_iter):
        pass


if __name__ == '__main__':
    Train()
#