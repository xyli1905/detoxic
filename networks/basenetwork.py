from utils import util
import torch
import torch.nn.functional as F
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
