# for model training
from networks.LR import BoW, EmbBoW, EmbLR
from options.base_options import BaseOption
from utils import util
import torch
from torch.utils.data import DataLoader
import pickle
import time
import os

class Train:
    def __init__(self, model, train_data):
        # parse param & options & setup
        self._opt = BaseOption().parse()
        self._start_epoch = 0
        self._training_size = train_data.shape[0]
        ##directory for saving/loading optimizer
        self._OPT_dir = os.path.join(self._opt.chkp_dir, self._opt.model_type)
        
        # NOTE: loading method will be changed in the future
        #load training dataset
        #self._training_dataset = 
        #
        # data processing for training
        #self._dataset = train_data
        self._data_train = DataLoader(train_data, batch_size=self._opt.batch_size,
                                      shuffle=True, drop_last=False)

        # setup model & training
        self._model = model
        # self._model = model(self._opt)
        self._lossfun = torch.nn.CrossEntropyLoss()
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=self._opt.lr)

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
        # may initialize info of training here
        # e.g. self._total_step
        # ...
        if self._start_epoch > 0:
            assert self._start_epoch < self._opt.max_epoch, \
                   "loaded epoch >= max_epoch, nothing to train"
            self._start_epoch -= 1

        for i_epoch in range(self._start_epoch, self._opt.max_epoch):
            epoch_time_start = time.time()

            # train epoch
            print("Start epoch %d / %d, \t at %s" % \
                  (i_epoch+1, self._opt.max_epoch, time.asctime()))
            self._train_epoch(i_epoch)

            # save model after each epoch here, note i_epoch is 0-based
            if (i_epoch+1) % self._opt.save_freq == 0 or (i_epoch+1) == self._opt.max_epoch:
                self._model.save(i_epoch+1)
                self._save_opt(i_epoch+1)

            # training time for each epoch
            time_cost = time.time() - epoch_time_start
            print("End of epoch %d / %d \t Time taken: %d sec (or % d min)\n" % \
                  (i_epoch+1, self._opt.max_epoch, time_cost, time_cost / 60.))

            # may updata learning rate here
            # if i_epoch > ...

    def _train_epoch(self, i_epoch):
        # display records for the number of trained iters
        print(" 0%s"%("%"), end='')
        p = 0.1

        for i_train_batch, train_batch in enumerate(self._data_train):
            
            # forward model
            y_pred = self._model.forward(train_batch[:, :-1])
            LRloss = self._lossfun(y_pred, train_batch[:, -1])
            
            # backprop
            self._optimizer.zero_grad()
            LRloss.backward()
            self._optimizer.step()

            # update & display progress
            if (i_train_batch+1)*self._opt.batch_size >= p*self._training_size:
                print(" - %.0f%s" %(100.*p,"%"), end='')
                p += 0.1

        # end of display progress
        print(" end")

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


def main():
    with open('./data_proc/processed_data/train_mat.pkl', 'rb') as f:
        train_data = pickle.load(f)
    Train(EmbLR, train_data)


if __name__ == '__main__':
    main()
#
