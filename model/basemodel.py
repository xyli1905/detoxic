# model prototype
from networks.LR import BoW, EmbBoW, EmbLR
from networks.GRU import GRULayer
from networks.LSTM import LSTMLayer
from utils.util import save_param, load_param
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import pickle


class BaseModel:
    def __init__(self, opt):
        self._name = "BaseModel"
        self._net_dict = {"BoW": BoW, "EmbBoW": EmbBoW, "EmbLR": EmbLR, "GRU": GRULayer, "LSTM": LSTMLayer}
        self._opt = opt
        self._set_model_directory()
        self._is_train = opt.is_train
        self.model_epoch = 0
        self._accum_loss = 0.
        self._eta = 1.


    def set_input(self, data):
        raise NotImplementedError('set_input not implemented for BaseModel')

    def forward(self, input):
        raise NotImplementedError('forword not implemented for BaseModel')

    def update_parameters(self, input):
        raise NotImplementedError('update_parameters not implemented for BaseModel')

    def update_learning_rate(self):
        raise NotImplementedError('update_learning_rate not implemented for BaseModel')
    

    def predict(self, idx_seq):
        # 0: sincere; 1: toxic
        y = F.softmax(self.forward(idx_seq), dim=1)
        y_pred = torch.argmax(y, dim=1)
        return y_pred

    def evaluate(self, test_data):
        '''
        assuming test_data's last col is label
        '''
        total_num = test_data.shape[0]
        eval_data = DataLoader(test_data, 
                                batch_size=1000,
                                shuffle=False,
                                drop_last=False)
        A  = 0.
        B  = 0.
        # C  = 0.
        FP = 0.
        FN = 0.
        for i_batch, eval_batch in enumerate(eval_data):

            batch_pred = self.predict(eval_batch[:,:-1])
            batch_diff = batch_pred - torch.squeeze(eval_batch[:,-1])
            batch_diff_pos = batch_diff[batch_diff ==  1]
            batch_diff_neg = batch_diff[batch_diff == -1]

            A += torch.sum(batch_pred) #pred P
            B += torch.sum(eval_batch[:,-1]) #actual P
            # C += torch.sum(torch.abs(batch_diff))
            FP += torch.sum(batch_diff_pos)
            FN -= torch.sum(batch_diff_neg)

        fA  = float(A)
        fB  = float(B)
        fFP = float(FP)
        fFN = float(FN)
        fC  = fFP + fFN
        
        TP = (fA+fB-fC)/2.

        f1score   = 2.*TP/(fA+fB)
        recall    = TP/fB
        precision = TP/fA if fA > 0. else -1.
        accuracy  = (1. - fC/float(total_num))*100.

        print("Predicted Positive: %d" % (fA))
        print("   Actual Positive: %d" % (fB))
        print(" Incorrect Predict: %d(total) %d(FP) %d(FN)" % (fC, fFP, fFN))
        print("\n F1 Score: %.5f" % (f1score))
        print("   Recall: %.5f" % (recall))
        print("Precision: %.5f" % (precision))
        print(" Accuracy: %.5f%s" % (accuracy, "%"))


    def save(self, epoch_idx):
        # self._save_network(epoch_idx)
        # self._save_optimizer(epoch_idx)
        raise NotImplementedError('save not implemented for BaseModel')

    def load(self, epoch_idx):
        # self._load_network(epoch_idx)
        # self._load_optimizer(epoch_idx)
        raise NotImplementedError('load not implemented for BaseModel')


    def _set_model_directory(self):
        self._model_dir = os.path.join(self._opt.chkp_dir, self._opt.model_type)

    def _save_network(self, network, name, epoch_idx):
        '''
        save trained network parameters
        '''
        save_param(network, self._model_dir, "network", name, epoch_idx)

    def _load_network(self, network, name, epoch_idx=-1):
        '''
        load trained network with epoch_idx,
        if idx=-1 load the one with max idx
        '''
        loaded_idx = load_param(network, self._model_dir, "network", name, epoch_idx)
        self.model_epoch = loaded_idx

    def _save_optimizer(self, optimizer, name, epoch_idx):
        '''
        save optimizer parameters
        '''
        save_param(optimizer, self._model_dir, "optimizer", name, epoch_idx)

    def _load_optimizer(self, optimizer, name, epoch_idx=-1):
        '''
        load optimizer with epoch_idx,
        if idx=-1 load the one with max idx
        '''
        loaded_idx =load_param(optimizer, self._model_dir, "optimizer", name, epoch_idx)
        self.model_epoch = loaded_idx
    