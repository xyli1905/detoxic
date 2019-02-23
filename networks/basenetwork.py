from utils import util
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle
import os

class BaseNetwork(torch.nn.Module):
    def __init__(self, opt):
        super(BaseNetwork, self).__init__()
        self.name = "BaseNetwork"
        self._opt = opt
        self._save_times = 0
        self.trained = False
        ##directory for saving/loading networks
        self._NET_dir = os.path.join(self._opt.chkp_dir, self._opt.model_type)

    def forward(self, input):
        assert False, "set_input not implemented"

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
        A = 0.
        B = 0.
        C = 0.
        for i_batch, eval_batch in enumerate(eval_data):

            batch_pred = self.predict(eval_batch[:,:-1])
            batch_diff = batch_pred - torch.squeeze(eval_batch[:,-1])

            A += torch.sum(batch_pred)
            B += torch.sum(eval_batch[:,-1])
            C += torch.sum(torch.abs(batch_diff))

        fA = float(A)
        fB = float(B)
        fC = float(C)
        TP = (fA+fB-fC)/2.

        f1score   = 2.*TP/(fA+fB)
        recall    = TP/fB
        precision = TP/fA
        accuracy  = (1. - fC/float(total_num))*100.

        print("Predicted Positive: %d" % (fA))
        print("   Actual Positive: %d" % (fB))
        print(" Incorrect Predict: %d" % (fC))
        print("\n F1 Score: %.5f" % (f1score))
        print("   Recall: %.5f" % (recall))
        print("Precision: %.5f" % (precision))
        print(" Accuracy: %.5f%s" % (accuracy, "%"))

    def save(self, epoch_idx):
        '''
        save trained parameters
        '''
        fname = 'net_{}_{}_id.pth'.format(self.name, str(epoch_idx))
        self._save_times += 1
        save_path = os.path.join(self._NET_dir, fname)
        torch.save(self.state_dict(), save_path)
        print(" model %s saved at %s" % (self.name, save_path))
        
    def load(self, idx=-1):
        '''
        load trained model with idx,
        if idx=-1 load the one with max idx
        '''
        net_mark = 'net_{}'.format(self.name)
        file_path, _ = util.find_file(net_mark, self._NET_dir, idx)

        self.load_state_dict(torch.load(file_path))
        print(" model %s loaded from %s" % (self.name, file_path))

#
