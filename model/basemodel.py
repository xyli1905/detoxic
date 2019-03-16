# model prototype
from networks.LR import BoW, EmbBoW, EmbLR
from networks.GRU import GRUnet
from networks.LSTM import LSTMnet
from utils.util import save_param, load_param
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
import pandas as pd


class BaseModel:
    def __init__(self, opt):
        self._name = "BaseModel"
        self._net_dict = {"BoW": BoW, "EmbBoW": EmbBoW, "EmbLR": EmbLR,
                          "GRU": GRUnet, "LSTM": LSTMnet}
        self._opt = opt
        self._set_model_directory()
        self._is_train = opt.is_train
        self.model_epoch = 0
        self._accum_loss = 0.
        self._eta = 1.


    def set_input(self, data):
        raise NotImplementedError('set_input not implemented')

    def forward(self, input):
        raise NotImplementedError('forword not implemented')

    def update_parameters(self, input):
        raise NotImplementedError('update_parameters not implemented')

    def update_learning_rate(self):
        raise NotImplementedError('update_learning_rate not implemented')
    

    def predict(self, idx_seq):
        '''
        # 0: sincere; 1: toxic
        presently threshold = 0.5
        '''
        with torch.no_grad():
            y = F.softmax(self.forward(idx_seq), dim=1)
            y_pred = y[:,1] >= self._opt.threshold
            # y_pred = torch.argmax(y, dim=1)

        return y_pred.long()

    def evaluate(self, test_data):
        '''
        assuming test_data's last col is label
        '''
        total_num = test_data.shape[0]
        eval_data = DataLoader(test_data, 
                                batch_size=1000,
                                shuffle=False,
                                drop_last=False)

        # if debug initialize np arraies to recored wrong pred
        if self._opt.is_debug:
            FP_mark = 1
            FN_mark = -1
            FP_idx = np.empty((0,1), np.int)
            FN_idx = np.empty((0,1), np.int)

        predP   = 0.
        FP = 0.
        FN = 0.
        for i_batch, eval_batch in enumerate(eval_data):

            batch_pred = self.predict(eval_batch[:,:-1])
            batch_diff = batch_pred - torch.squeeze(eval_batch[:,-1])
            batch_diff_pos = batch_diff[batch_diff ==  1]
            batch_diff_neg = batch_diff[batch_diff == -1]

            # if debug, gradually record the fail case indices
            if self._opt.is_debug:
                # print((batch_diff==FP_mark).nonzero().numpy() + i_batch*1000)
                tmp_FP = (batch_diff==FP_mark).nonzero().numpy() + i_batch*1000
                tmp_FN = (batch_diff==FN_mark).nonzero().numpy() + i_batch*1000
                FP_idx = np.vstack((FP_idx, tmp_FP))
                FN_idx = np.vstack((FN_idx, tmp_FN))

            predP += torch.sum(batch_pred)
            FP += torch.sum(batch_diff_pos)
            FN -= torch.sum(batch_diff_neg)

        # if debug, dump the wrong pred idx for analyze
        if self._opt.is_debug:
            self._dump_falsepred_idx(FP_idx, FN_idx)

        TP = predP - FP
        TN = total_num - predP - FN

        # screen print & save
        self._print_confusion_mat(TP, FP, FN, TN)
        self._print_and_save_metric(TP, FP, FN, TN)
        
    def _print_confusion_mat(self, TP, FP, FN, TN):
        sTP = "{}(TP)".format(str(int(TP)))
        sFP = "{}(FP)".format(str(int(FP)))
        sTN = "{}(TN)".format(str(int(TN)))
        sFN = "{}(FN)".format(str(int(FN)))

        print("Confusion Matrix:")
        print("{:>11}| {:^12} {:^12}".format("  ","Actual Pos", "Actual Neg"))
        print("{:>11}|{}".format(4*"- ", 15*"- "))
        print("{:>11}| {:^12} {:^12}".format("Pred Pos", sTP, sFP))
        print("{:>11}| {:^12} {:^12}".format("Pred Neg", sFN, sTN))

    def _print_and_save_metric(self, TP, FP, FN, TN):
        fTP = float(TP)
        fFP = float(FP)
        fTN = float(TN)
        fFN = float(FN)

        f1score   = 2.*fTP/(2.*fTP+fFP+fFN)
        recall    = fTP/(fFN+fTP)
        precision = fTP/(fTP+fFP) if (fTP+fFP) > 0. else -1.
        fallout   = fFP/(fFP+fTN)
        accuracy  = 100.*(fTP+fTN)/(fTP+fFP+fTN+fFN)

        print("\n F1 Score: %.5f" % (f1score))
        print("   Recall: %.5f" % (recall))
        print("Precision: %.5f" % (precision))
        print("  Fallout: %.5f" % (fallout))
        print(" Accuracy: %.5f%s" % (accuracy, "%"))
        print("")

        # save to csv file for easy use in the future
        self._save_metric(fTP,fFP,fTN,fFN,f1score,recall,precision,fallout,accuracy)
        
    def _save_metric(self, fTP,fFP,fTN,fFN,f1score,recall,precision,fallout,accuracy):
        Dict = {"threshold": self._opt.threshold,
                "TP": fTP,
                "FP": fFP,
                "TN": fTN,
                "FN": fFN,
                "F1": f1score,
                "Recall": recall,
                "Precision": precision,
                "Fallout": fallout,
                "Accuracy": accuracy/100.
               }
        
        fname = "met_{}_{}.csv".format(str(self._opt.classifier_net), str(self._opt.tag))
        fpath = os.path.join(self._opt.results_dir, fname)

        if os.path.isfile(fpath):
            df_old = pd.read_csv(fpath, index_col=0)
            df_new = pd.DataFrame.from_dict(Dict, orient="index", columns=["epoch_"+str(self.model_epoch)])
            df = pd.concat([df_old, df_new], axis = 1, sort=False)
        else:
            df = pd.DataFrame.from_dict(Dict, orient="index", columns=["epoch_"+str(self.model_epoch)])
        df.to_csv(fpath)

        print("metric for epoch %d saved/add to %s\n" % (self.model_epoch, fpath))

    def _dump_falsepred_idx(self, FP_idx, FN_idx):
        F_dict = {"FP": FP_idx,
                  "FN": FN_idx
                 }

        fname = "falsepredidx.pkl"
        fpath = os.path.join(self._opt.debug_dir, fname)

        with open(fpath, 'wb') as f:
            pickle.dump(F_dict, f)


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
#