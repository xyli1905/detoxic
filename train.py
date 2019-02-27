# for model training
from options.base_options import BaseOption
from utils import util
from utils.dataloader import CustomDataLoader
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
    def __init__(self, **jupyter_input):
        # parse param & options & setup
        self._opt = BaseOption().parse()
        self._start_epoch = 0
        ##directory for saving/loading optimizer
        self._OPT_dir = os.path.join(self._opt.chkp_dir, self._opt.model_type)

        # jupyter parser is for easy use in jupyter notebook, 
        # -- assuming has at least model & train_data as inputs
        if len(jupyter_input) != 0:
            self._model, self._dataset, max_epoch, load_epoch_idx = util.jupyter_parser(jupyter_input)
            if max_epoch > 0:
                self._opt.max_epoch = max_epoch
            if load_epoch_idx > 0:
                self._opt.load_epoch_idx = load_epoch_idx
        else:
            self._dataset = util.load_training_data(self._opt)
            self._model = util.set_model(self._opt)
        if self._opt.valid_num > 0:
            self._dataset = self._dataset[:-self._opt.valid_num, :]
        self._training_size = self._dataset.shape[0]

        # setup loss function and optimizer
        if self._opt.triplet:
            self._lossfun = torch.nn.TripletMarginLoss(margin=self._opt.margin, p=2)
        else:
            self._lossfun = torch.nn.CrossEntropyLoss()
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=self._opt.lr)

        # Initialize data loader
        self._dataloader = CustomDataLoader(self._dataset, self._opt)

        # if load_epoch > 0 try load existing model
        # & if -1 load the ones with maximum existing epoch idx
        if self._opt.load_epoch_idx > 0 or self._opt.load_epoch_idx == -1:
            self._load_epoch(self._opt.load_epoch_idx)

        # train model
        self._train()

        # update training status
        if not self._model.trained:
            self._model.trained = True

    def _train(self):
        print("\nStart training model: \n%s" % (str(self._model)))
        # condition for continued training
        if self._start_epoch > 0:
            assert self._start_epoch < self._opt.max_epoch, \
                   "loaded epoch >= max_epoch, nothing to train"

        # if train in normal way; i.e. not use triplet loss
        if not self._opt.triplet:
            self._data_train = self._dataloader.load_batches()

        # Main loop, train each epoch
        for i_epoch in range(self._start_epoch, self._opt.max_epoch):
            epoch_time_start = time.time()

            # if use triplet loss re-sample triplet for every epoch
            if self._opt.triplet:
                self._data_train = self._dataloader.load_triplets()

            # train epoch
            print("\nStart epoch %d / %d, \t at %s" % \
                  (i_epoch+1, self._opt.max_epoch, time.asctime()))
            self._train_epoch(i_epoch)

            # save model after each epoch here, note i_epoch is 0-based
            if (i_epoch+1) % self._opt.save_freq == 0 or (i_epoch+1) == self._opt.max_epoch:
                self._model.save(i_epoch+1)
                self._save_opt(i_epoch+1)

            # training time for each epoch
            time_cost = time.time() - epoch_time_start
            print("End of epoch %d / %d \t Time taken: %d sec (or % d min)" % \
                  (i_epoch+1, self._opt.max_epoch, time_cost, time_cost / 60.))

            # updata learning rate
            # if i_epoch > ...

    def _train_epoch(self, i_epoch):
        # display records for the number of trained iters
        print(" 0%s"%("%"), end='')
        p = 0.1

        for i_train_batch, train_batch in enumerate(self._data_train):
            
            # forward model
            if self._opt.triplet:
                # presently in LR&GRU encoding has dim 64
                anchor = self._model.forward(train_batch["anchor"][:,:-1], use_encoding=True)
                positive = self._model.forward(train_batch["positive"][:,:-1], use_encoding=True)
                negative = self._model.forward(train_batch["negative"][:,:-1], use_encoding=True)
                Loss = self._lossfun(anchor, positive, negative)
            else:
                y_pred = self._model.forward(train_batch[:, :-1])
                Loss = self._lossfun(y_pred, train_batch[:, -1])
            
            # backprop
            self._optimizer.zero_grad()
            Loss.backward()
            self._optimizer.step()

            # update & display progress
            if (i_train_batch+1)*self._opt.batch_size >= p*self._training_size:
                print(" - %.0f%s" %(100.*p,"%"), end='')
                p += 0.1

        # end of display progress
        print(" done")

    def _display_progress(self):
        pass

    def _save_opt(self, idx):
        '''
        idx is from model.save, equal to model.training_times
        save optimizer parameters
        '''
        fname = 'opt_{}_{}_id.pth'.format(self._model.name, str(idx))
        save_path = os.path.join(self._OPT_dir, fname)
        torch.save(self._optimizer.state_dict(), save_path)
        print(" optimizer for %s saved at %s" % (self._model.name, save_path))

    def _load_opt(self, idx):
        '''
        load optimizer
        if idx=-1 load the one with max idx
        '''
        opt_mark = 'opt_{}'.format(self._model.name)
        file_path, idx_num = util.find_file(opt_mark, self._OPT_dir, idx)

        # self._start_epoch is 0-based, idx_num is 1-based
        # will do self._start_epoch later
        self._start_epoch = idx_num
        
        self._optimizer.load_state_dict(torch.load(file_path))
        print(" optimizer for %s loaded from %s" % (self._model.name, file_path))

    def _load_epoch(self, idx=-1):
        '''
        load both exisiting model & its optimizer,
        continue the training.
        if idx=-1 load the ones with max idx
        '''
        self._model.load(idx)
        self._load_opt(idx)


if __name__ == '__main__':
    Train()
#
