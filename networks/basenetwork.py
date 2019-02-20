import torch
import torch.nn.functional as F
import pickle
import os

class BaseNetwork(torch.nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()
        self.name = "BaseNetwork"
        self._chkp_dir = "/Users/xyli1905/Projects/NLP/detoxic/checkpoints"
        self._save_times = 0

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
        save_path = os.path.join(self._chkp_dir, fname)
        torch.save(self.state_dict(), save_path)
        print("Model %s saved at %s" % (self.name, save_path))
        
    def load(self, idx=-1):
        '''
        load trained model with idx,
        if idx=-1 load the one with max idx
        '''
        file_path = self._find_net_file(idx)

        self.load_state_dict(torch.load(file_path))
        print("Model %s loaded from %s" % (self.name, file_path))

    def _find_net_file(self, idx):
        '''
        return the model file for self._model.name & given idx
        if idx=-1 load the one with max idx
        '''
        net_mark = 'net_{}'.format(self.name)

        if idx == -1:
            idx_num = -1
            for file in os.listdir(self._chkp_dir):
                if file.startswith(net_mark):
                    idx_num = max(idx_num, int(file.split('_')[2]))
            assert idx_num >= 0, 'Model %s file not found' % self.name
        else:
            found = False
            for file in os.listdir(self._chkp_dir):
                if file.startswith(net_mark):
                    found = int(file.split('_')[2]) == idx
                    if found: break
            assert found, 'Model %s file for epoch %i not found' % (self.name, idx)
            idx_num = idx

        fname = 'net_{}_{}_id.pth'.format(self.name, str(idx_num))
        file_path = os.path.join(self._chkp_dir, fname)

        return file_path
#
