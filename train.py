# for model training
from networks.LR import BoW, EmbBoW, EmbLR
import torch
from torch.utils.data import DataLoader
import time
import os

class Train:
    def __init__(self, model, train_data, max_epoch=5, load_epoch_idx=0):
        self._chkp_dir = "/Users/xyli1905/Projects/NLP/detoxic/checkpoints"
        # parameter
        self._max_epoch = max_epoch
        self._batch_size = 64
        self.save_freq = 2
        self._start_epoch = 0
        
        # data processing for training
        #self._dataset = train_data
        self._dataset_train = DataLoader(train_data, batch_size=self._batch_size,
                                         shuffle=True, drop_last=True)

        # define model
        self._model = model
        self._lossfun = torch.nn.CrossEntropyLoss()
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=0.01)
        
        # if load_epoch > 0 try load existing model
        # & if -1 load the ones with maximum existing epoch idx
        if load_epoch_idx > 0 or load_epoch_idx == -1:
            self._load_epoch(load_epoch_idx)

        # train model
        self._train()
        
        # update training flag
        if self._model.trained:
            self._model.training_times += 1
        else:
            self._model.trained = True
            self._model.training_times = 1
    
    def _train(self):
        # may initialize info of training here
        # e.g. self._total_step
        # ...
        if self._start_epoch > 0:
            assert self._start_epoch < self._max_epoch, \
                   "loaded epoch >= max_epoch, nothing to train"
            self._start_epoch -= 1

        for i_epoch in range(self._start_epoch, self._max_epoch):
            epoch_time_start = time.time()
            
            # train epoch
            print("Start epoch %d / %d, \t at %s" % \
                  (i_epoch+1, self._max_epoch, time.asctime()))
            self._train_epoch(i_epoch)
        
            # save model after each epoch here, note i_epoch is 0-based
            if (i_epoch+1) % self.save_freq == 0 or (i_epoch+1) == self._max_epoch:
                self._model.save(i_epoch+1)
                self._save_opt(i_epoch+1)
            
            # training time for each epoch
            time_cost = time.time() - epoch_time_start
            print("End of epoch %d / %d \t Time taken: %d sec (or % d min)" % \
                  (i_epoch+1, self._max_epoch, time_cost, time_cost / 60.))
            
            # may updata learning rate here
            # if i_epoch > ...

    def _train_epoch(self, i_epoch):

        for i_train_batch, train_batch in enumerate(self._dataset_train):
            
            # forward model
            y_pred = self._model.forward(train_batch[:, :-1])
            LRloss = self._lossfun(y_pred, train_batch[:, -1])
            
            # backprop
            self._optimizer.zero_grad()
            LRloss.backward()
            self._optimizer.step()
            
    def _save_opt(self, idx):
        '''
        idx is from model.save, equal to model.training_times
        save optimizer parameters
        '''
        fname = 'opt_{}_{}_id.pth'.format(self._model.name, str(idx))
        save_path = os.path.join(self._chkp_dir, fname)
        torch.save(self._optimizer.state_dict(), save_path)
        print("optimizer for %s saved at %s" % (self._model.name, save_path))
    
    def _find_opt_file(self, idx):
        '''
        return the opt file for self._model.name & given idx
        if idx=-1 load the one with max idx
        '''
        opt_mark = 'opt_{}'.format(self._model.name)

        if idx == -1:
            idx_num = -1
            for file in os.listdir(self._chkp_dir):
                if file.startswith(opt_mark):
                    idx_num = max(idx_num, int(file.split('_')[2]))
            assert idx_num >= 0, 'opt file not found'
        else:
            found = False
            for file in os.listdir(self._chkp_dir):
                if file.startswith(opt_mark):
                    found = int(file.split('_')[2]) == idx
                    if found: break
            assert found, 'opt file for epoch %i not found' % idx
            idx_num = idx

        # self._start_epoch is 0-based, idx_num is 1-based
        # will do self._start_epoch later
        self._start_epoch = idx_num

        fname = 'opt_{}_{}_id.pth'.format(self._model.name, str(idx_num))
        file_path = os.path.join(self._chkp_dir, fname)

        return file_path

    def _load_opt(self, idx):
        '''
        load optimizer
        if idx=-1 load the one with max idx
        '''
        file_path = self._find_opt_file(idx)
        
        self._optimizer.load_state_dict(torch.load(file_path))
        print("optimizer for %s loaded from %s" % (self._model.name, file_path))

    def _load_epoch(self, idx=-1):
        '''
        load both exisiting model & its optimizer,
        continue the training.
        if idx=-1 load the ones with max idx
        '''
        self._model.load(idx)
        self._load_opt(idx)


def main():
    model = EmbLR()
    Train(model, max_epoch=10)


if __name__ == '__main__':
    main()
#
